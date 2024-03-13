import org.apache.spark.ml.feature.{OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.sql.{DataFrame, Dataset, Encoders, SparkSession}
import org.apache.spark.sql.functions.{avg, col, corr, count, desc, expr, isnan, when}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator}

object MachineLearningAssignment {

  case class TitanicData(
                          PassengerId: Long,
                          Survived: Int,
                          Pclass: Int,
                          Name: String,
                          Sex: String,
                          Age: Int,
                          SibSp: Int,
                          Parch: Int,
                          Ticket: String,
                          Fare: Double,
                          Cabin: String,
                          Embarked: String
                        )

  case class TitanicTestData(
                          PassengerId: Long,
                          Pclass: Int,
                          Name: String,
                          Sex: String,
                          Age: Int,
                          SibSp: Int,
                          Parch: Int,
                          Ticket: String,
                          Fare: Double,
                          Cabin: String,
                          Embarked: String
                        )

  def main(args: Array[String]) {
    val spark = SparkSession.builder
      .appName("MachineLearningAssignment")
      .getOrCreate()

    import spark.implicits._
    val schema = Encoders.product[TitanicData].schema
    val filePath = "src/main/resources/train.csv"
    val trainDS: Dataset[TitanicData] = spark.read.option("header", "true").schema(schema).csv(filePath).as[TitanicData]

    trainDS.describe().show()
    trainDS.printSchema()

    // Exploratory Data Analysis
    val familySizeDF = trainDS.withColumn("FamilySize", col("SibSp") + col("Parch"))
    val familySizeDistribution = familySizeDF.groupBy("FamilySize").count()
    familySizeDistribution.show()
    val correlationDF = trainDS.select(corr("Age", "Fare").alias("Age_Fare_Correlation"))
    correlationDF.show()

    val correlation2DF = familySizeDF.select(corr("Age", "FamilySize").alias("Age_FamilySize_Correlation"))
    correlation2DF.show()

    // Counting null or NaN values in each column
    val nullCounts = trainDS.columns.map(c => count(when(isnan(col(c)) || col(c).isNull, c)).alias(c))
    trainDS.select(nullCounts: _*).show()

    // Handling missing values in Age column
    val avgAgeByClass = trainDS.groupBy("Pclass").agg(avg("Age").alias("average_age"))
    val dfWithAge = trainDS.join(avgAgeByClass, Seq("Pclass"), "left_outer")
    val dfNew = dfWithAge.withColumn("Age", when(col("Age").isNull, col("average_age")).otherwise(col("Age"))).drop("average_age")

    // Feature extraction
    val dfWithOutNull = dfNew.drop("Cabin", "Ticket", "Embarked")

    //Creating new attributes
    val enhancedTrainDS = dfWithOutNull.withColumn("FamilySize", col("SibSp") + col("Parch"))

    // Numerical features with continuous values - Age, Fare, FamilySize
    val numFeatColNames = Seq("Age", "FamilySize", "Fare")
    val assembler = new VectorAssembler().setInputCols(numFeatColNames.toArray).setOutputCol("numerical-features")
    val assembledDF = assembler.transform(enhancedTrainDS)

    // Scaling numerical features
    val scaler = new StandardScaler().setInputCol("numerical-features").setOutputCol("scaled-numerical-features").setWithStd(true).setWithMean(true)
    val scalerModel = scaler.fit(assembledDF)
    val scaledDF = scalerModel.transform(assembledDF)

    // Handling categorical features - Gender and Pclass
    val genderIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex").fit(scaledDF)
    val pClassEncoder = new OneHotEncoder().setInputCol("Pclass").setOutputCol("PclassEncoded").fit(scaledDF)

    // Assembling final features vector
    val finalAssembler = new VectorAssembler().setInputCols(Array("scaled-numerical-features", "SexIndex", "PclassEncoded")).setOutputCol("features")
    val finalAssembledDF = finalAssembler.transform(pClassEncoder.transform(genderIndexer.transform(scaledDF)))

    // Splitting data into training and validation sets
    val Array(trainingData, validationData) = finalAssembledDF.randomSplit(Array(0.8, 0.2), seed = 1234)

    // Creating and training Random Forest classifier
    val randomForest = new RandomForestClassifier().setLabelCol("Survived").setFeaturesCol("features")
    val pipeline = new Pipeline().setStages(Array(randomForest))
    val model = pipeline.fit(trainingData)

    // Making predictions on validation set
    val predictions = model.transform(validationData)

    // Evaluating accuracy on validation Set
    val evaluatorAccuracy = new BinaryClassificationEvaluator().setLabelCol("Survived").setRawPredictionCol("rawPrediction").setMetricName("areaUnderROC")
    val accuracy = evaluatorAccuracy.evaluate(predictions)
    println(s"Validation Accuracy: ${accuracy * 100}%")

    //Loading the test data



    val testFilePath = "src/main/resources/test.csv"
    val schemaTest = Encoders.product[TitanicTestData].schema
    val testData: Dataset[TitanicTestData] = spark.read.option("header", "true").schema(schemaTest).csv(testFilePath).as[TitanicTestData]

    //preparing the test data with filling the missing values function and adding the new column Family Size
    val avgAgeByClassTest = testData.groupBy("Pclass").agg(avg("Age").alias("average_age"))
    val dfWithAgeTest = testData.join(avgAgeByClassTest, Seq("Pclass"), "left_outer")
    val dfNewTest = dfWithAgeTest.withColumn("Age", when(col("Age").isNull, col("average_age")).otherwise(col("Age"))).drop("average_age")

    // Feature extraction
    val dfWithOutNullTest = dfNewTest.drop("Cabin", "Ticket", "Embarked")

    //Creating new attributes
    val enhancedTrainDSTest = dfWithOutNullTest.withColumn("FamilySize", col("SibSp") + col("Parch"))
    // Apply preprocessing steps to test data
    val finalDataTest = enhancedTrainDSTest.na.drop()
    val testAssembledDF = finalAssembler.transform(pClassEncoder.transform(genderIndexer.transform(scalerModel.transform(assembler.transform(finalDataTest)))))


    val testPredictions = model.transform(testAssembledDF)

    // Evaluate model performance on test data
    val result = testPredictions.select("PassengerId", "prediction")
    result.coalesce(1).
      write.option("header", "true").
      csv("src/main/resources/result")

    spark.stop()
  }
}
