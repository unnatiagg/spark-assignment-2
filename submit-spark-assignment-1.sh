#!/bin/bash

echo "Compiling and assembling application..."
sbt assembly

# Directory where spark-submit is defined
# Install spark from https://spark.apache.org/downloads.html
# NOTE: you need to change this to your spark installation

# Replace "home" with "Users" in MacOS
SPARK_HOME=/opt/homebrew/Cellar/apache-spark/3.5.0

# JAR containing a assignment-1
JARFILE=`pwd`/target/scala-2.12/machinelearningassignment-assembly-0.1.0-SNAPSHOT.jar

# Run it locally
${SPARK_HOME}/bin/spark-submit --class MachineLearningAssignment --master local $JARFILE
