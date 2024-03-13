ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.18"

ThisBuild / libraryDependencies += "org.apache.spark" %% "spark-core" % "3.5.0" % "provided"
ThisBuild / libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.5.0" % "provided"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.5.0" % "provided"

lazy val root = (project in file("."))
  .settings(
    name := "MachineLearningAssignment"
  )


