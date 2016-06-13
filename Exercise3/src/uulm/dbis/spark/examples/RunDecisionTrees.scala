package uulm.dbis.spark.examples


import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._

import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model._

import org.apache.spark.rdd._
import org.apache.spark.mllib.evaluation._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._

object RunDecistionTrees {
  def main(args: Array[String]) {
  
  val sparkConf = new SparkConf().setAppName("RunDecisionTrees")
  val sc = new SparkContext(sparkConf)                           // Create SparkContext
  val rawData = sc.textFile("./covtype.data")
  val data = rawData.map { line =>
  val values = line.split(',').map(_.toDouble)
  val featureVector = Vectors.dense(values.init)
  val label = values.last - 1
  LabeledPoint(label, featureVector)
}

val Array(trainData, valData, testData) =
  data.randomSplit(Array(0.8, 0.1, 0.1))
trainData.cache()
valData.cache()
testData.cache()

val model = DecisionTree.trainClassifier(
  trainData, 7, Map[Int,Int](), "gini", 4, 100)

def getMetrics(model: DecisionTreeModel, data: 	RDD[LabeledPoint]):
  MulticlassMetrics = {
    val predictionsAndLabels = data.map(example =>
      (model.predict(example.features), example.label)
    )
    new MulticlassMetrics(predictionsAndLabels)
  }

val metrics = getMetrics(model, valData)

(0 until 7).map(
  label => (metrics.precision(label), metrics.recall(label))
).foreach(println)

def classProbabilities(data: RDD[LabeledPoint]): Array[Double] = {
  val countsByCategory = data.map(_.label).countByValue()
  val counts = countsByCategory.toArray.sortBy(_._1).map(_._2)
  counts.map(_.toDouble / counts.sum)
}

val trainPriorProbabilities = classProbabilities(trainData)
val valPriorProbabilities = classProbabilities(valData)
trainPriorProbabilities.zip(valPriorProbabilities).map {
  case (trainProb, valProb) => trainProb * valProb
}.sum

val evaluations =
  for (impurity <- Array("gini", "entropy");
    depth <- Array(10, 20, 30);
    bins <- Array(50, 100, 300))
  yield {
    val model = DecisionTree.trainClassifier(
      trainData, 7, Map[Int,Int](), impurity, depth, bins)
    val predictionsAndLabels = valData.map(example =>
      (model.predict(example.features), example.label)
    )  
    val precision =
      new MulticlassMetrics(predictionsAndLabels).precision
    ((impurity, depth, bins), precision) }

evaluations.sortBy(_._2).reverse.foreach(println)

val data1 = rawData.map { line =>
  val values = line.split(',').map(_.toDouble)
  val wilderness = values.slice(10, 14).indexOf(1.0).toDouble
  val soil = values.slice(14, 54).indexOf(1.0).toDouble
  val featureVector =
    Vectors.dense(values.slice(0, 10) :+ wilderness :+ soil)
  val label = values.last - 1
  LabeledPoint(label, featureVector)
}
  
val Array(trainData1, valData1, testData1) =
  data.randomSplit(Array(0.8, 0.1, 0.1))

val evaluations2 =
  for (impurity <- Array("gini", "entropy");
    depth <- Array(20, 30);
    bins <- Array(100, 300))
  yield {
    val model = DecisionTree.trainClassifier(
      trainData, 7, Map(10 -> 4, 11 -> 40),
      impurity, depth, bins)
    val trainPrecision = getMetrics(model, trainData).precision
    val valPrecision = getMetrics(model, valData).precision
    ((impurity, depth, bins), (trainPrecision, valPrecision)) }
  
evaluations.sortBy(_._2).reverse.foreach(println)

val forest = RandomForest.trainClassifier(
  trainData.union(valData), 7, Map(10 -> 4, 11 -> 40), 20,
    "auto", "entropy", 30, 300)
    
val input = "2709,125,28,67,23,3224,253,207,61,6094,0,29"
val vector = Vectors.dense(input.split(',').map(_.toDouble))

forest.predict(vector)

}
}