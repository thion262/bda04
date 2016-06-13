

package uulm.dbis.spark.examples

import scala.collection.Map
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.recommendation._
import org.apache.spark.rdd.RDD

object Recommender {
  def main(args: Array[String]) {
    if (args.length < 2) {
      System.err.println("Correct usage: Cleaner <input-folder> <output-folder> <script-option>")
      System.exit(1)
    }
    
    val sparkConf = new SparkConf().setAppName("Cleaner")
    val sc = new SparkContext(sparkConf)                           // Create SparkContext
    
    val raw_artist = sc.textFile(args(0) + "/audio-data/short_artist.txt")
        raw_artist.persist()
    val raw_artistalias = sc.textFile(args(0) + "/audio-data/short_alias.txt")
        raw_artistalias.persist()
    val user_artist = sc.textFile(args(0) + "/audio-data/short_user.txt")
        user_artist.persist()
    
    var userArtist = user_artist.map{ line => 
      val tokens = line.split(' ') 
      (tokens(0).toInt, tokens(1).toInt, tokens(2).toInt) }
        
        
    var corruptedLinesArtistByID = sc.accumulator(0);
    var corruptedLinesAlias = sc.accumulator(0);
    
    // Clean artist_data.txt    
    var artistByID = raw_artist.flatMap { line => 
      val (id,name) = line.span(_ != '\t')
      if (name.isEmpty) { 
        corruptedLinesArtistByID += 1
        None
      }
      else {
        try {
          Some((id.toInt, name.trim))
        } catch {
          case e: NumberFormatException => None 
          }
      }
    }
    
    
    // Clean artist_alias.txt
    val artistAlias = raw_artistalias.flatMap { line =>
      val tokens = line.split('\t')
      if (tokens(0).isEmpty) { 
        corruptedLinesAlias += 1
        None  
      }
      else {
        Some((tokens(0).toInt,tokens(1).toInt))
      }
    }.collectAsMap()
    
    // Filter UNKOWN artists from artist_data
    // count lines that fit predicate
    val unknownCount = artistByID.collectAsMap().count(line =>
      (line._2.contains("unknown") && !(artistAlias.contains(line._1)))
    ).toString()
    
    // filter lines that fit predicate
    artistByID = artistByID.filter{ line =>
        !(line._2.contains("unknown") && !(artistAlias.contains(line._1)))
    } 
    
    // Pack counts into one String
    val builder = StringBuilder.newBuilder.append("Corrupted Lines in artist_data.txt: ").append(corruptedLinesArtistByID.value.toString())
    builder.append("\nCorrupted Lines in artist_alias.txt: ")
    builder.append(corruptedLinesAlias.value.toString())
    builder.append("\nFiltered Lines with unknown artists in artist_data.txt: ")
    builder.append(unknownCount)
    
    val countedParameters = builder.toString()
    sc.parallelize(Seq(countedParameters)).saveAsTextFile(args(1)+"/removedLines")
    
    // Histograms
    val bArtistAlias = sc.broadcast(artistAlias)
    
    val ua_abgeglichen = userArtist.map { line =>
      val finalArtistID = bArtistAlias.value.getOrElse(line._2, line._2)
      (line._1, finalArtistID, line._3)
    }
//    val ua_abgeglichenB = userArtist.flatMap{ line  =>                          // (UserID, ArtistID)
//      if(artistAlias.contains(line._2)) {                                      // Falls Artist in Alias enthalten
//        artistAlias.map( entry => 
//          if(entry._1 == line._2) {
//            (line._1, entry._2, line._3)                                       // BadID ersetzen
//          }
//          else if(entry._2 == line._2) {
//            (line._1, entry._2, line._3)                                       // BadID ersetzen
//          }
//          else {
//            (line._1, line._2, line._3)
//          })
//      }
//      else {
//        Some(line._1, line._2, line._3)                                        // Default
//      } 
//    }
    ua_abgeglichen.persist()
    
    val predA = ua_abgeglichen.map{ line => (line._1, line._2) }.countByKey()  // (UserID, #Artists)
    val predB = ua_abgeglichen.map{ line => (line._2, line._1) }.countByKey()  // (ArtistID, #Users)
    val predC = ua_abgeglichen.map{ line => (line._1, line._3) }.countByKey()  // (UserID, total count)
    
    val noofartists = sc.parallelize(predA.unzip._2.map(x => x.toDouble).toList)
    noofartists.histogram(10)
    val noofusers = sc.parallelize(predB.unzip._2.map(x => x.toDouble).toList)
    noofusers.histogram(10)
    
    // Filter Frequencies 1<x<100.000 and number of songs 50<x<100.000 
    // Collect artistIDs to filter out
    val filterArtistIDs = predB.flatMap(line =>
      if(line._2 == 1 || line._2 > 100000) {
        Some(line._1)
      }
      else {
        None
      }
    ).toList
    
    // Collect userIDs to filter out
    val filterUserIDs = predC.flatMap(line =>
      if(line._2 < 50 || line._2 > 100000) {
        Some(line._1)
      }
      else {
        None
      }
    ).toList
    
    // Filter out
     userArtist = userArtist.filter{line => !((filterArtistIDs.contains(line._2)) || (filterUserIDs.contains(line._1)))}
    
    // Whatever
    artistByID.saveAsTextFile(args(1) + "/cleaned_artist_data/")
    sc.parallelize(Seq(bArtistAlias.value)).saveAsTextFile(args(1) + "/cleaned_artist_alias/")
    userArtist.saveAsTextFile(args(1) + "/cleaned_user_artist/")
    
/*
 * PROBLEM 2	Evaluation of a Recommender System
 */
    
    //get ratings from cleaned user_artist_data (final artistIDs)
    val allData = ua_abgeglichen.map { line =>  
      Rating(line._1, line._2, line._3)
    }
    
    // Split the data into 5 Folds
    val split = allData.randomSplit(Array(0.2, 0.2, 0.2, 0.2, 0.2))

    // Set Script-Option, Initializations    
    var option = false
    var rank = List(10)
    var lambda = List(0.01)
    var alpha = List(1)
    
    if (args.length==3) { // Overwrite defaults
      option = true
      rank = List(10, 100)
      lambda = List(0.1, 10)
      alpha = List(15, 40)
    }
    var newbuilder = StringBuilder.newBuilder.append("Parameter-Trial-Option was: " + option.toString())
    
    // Compute the AUCs
    for(r <- rank) {
      for(l <- lambda) {
        for(a <- alpha) {
        val avgAucs = model(sc,allData,split,r,l,a)
        newbuilder.append("\n\nParameter(rank,lambda,alpha):\t" + r.toString() + ",\t" + l.toString() + ",\t" + a.toString())
        newbuilder.append("\nScore default predict:\t" + avgAucs._1.toString() + "\nScore own predict:\t" + avgAucs._2.toString())
        }
      }
    }
    
    // Save results of AUCs
    val meanAucStr = newbuilder.toString()
    sc.parallelize(Seq(meanAucStr)).coalesce(1, true).saveAsTextFile(args(1) + "/meanAUC")

    sc.stop()
  }

  def model(
      sc: SparkContext,
      allData: RDD[Rating],
      split: Array[RDD[Rating]],
      rank:Int,
      lambda:Double,
      alpha:Int): (Double,Double) = {
    

    // Initialize Array of 5 AUC values
    var aucValues = new Array[Double](split.length)
    var ownAucValues = new Array[Double](split.length)
    
    // Perform ALS 5 times
    for (i <- 0 until split.length) {
      var valid = split(i)
      var train = allData.subtract(valid)
      train.cache()
      valid.cache()

      val allItemIDs = allData.map(_.product).distinct().collect()
      val bAllItemIDs = sc.broadcast(allItemIDs)

      // implicit train with parameters: data, rank, iterations, lambda, alpha
      val model = ALS.trainImplicit(train, rank, 5, lambda, alpha)
      
      val auc = areaUnderCurve(valid, bAllItemIDs, model.predict)
      val aucOwn = areaUnderCurve(valid, bAllItemIDs, predictAvgListened(sc,train))
      aucValues(i) = auc
      ownAucValues(i) = aucOwn
      
      unpersist(model)
      train.unpersist()
      valid.unpersist()
    }

    // Calculate meanAUC
    val meanAuc = aucValues.sum / aucValues.size
    val meanOwnAuc = ownAucValues.sum / ownAucValues.size
    return (meanAuc,meanOwnAuc)
  }
  
    def predictAvgListened(sc: SparkContext, train: RDD[Rating])(allData: RDD[(Int,Int)]) = {
      val listencount = train.map(r => (r.product, r.rating)).groupByKey.mapValues(x => x.sum / x.size).collectAsMap()
      val bListenCount = sc.broadcast(listencount)
      allData.map { case (user, product) =>
      Rating(user, product, bListenCount.value.getOrElse(product, 0.0))
    }
  }
  
  // AUC method from given RunRecommender.scala
  def areaUnderCurve(
    positiveData: RDD[Rating],
    bAllItemIDs: Broadcast[Array[Int]],
    predictFunction: (RDD[(Int, Int)] => RDD[Rating])) = {
    // What this actually computes is AUC, per user. The result is actually something
    // that might be called "mean AUC".

    // Take held-out data as the "positive", and map to tuples
    val positiveUserProducts = positiveData.map(r => (r.user, r.product))
    // Make predictions for each of them, including a numeric score, and gather by user
    val positivePredictions = predictFunction(positiveUserProducts).groupBy(_.user)

    // BinaryClassificationMetrics.areaUnderROC is not used here since there are really lots of
    // small AUC problems, and it would be inefficient, when a direct computation is available.

    // Create a set of "negative" products for each user. These are randomly chosen
    // from among all of the other items, excluding those that are "positive" for the user.
    val negativeUserProducts = positiveUserProducts.groupByKey().mapPartitions {
      // mapPartitions operates on many (user,positive-items) pairs at once
      userIDAndPosItemIDs =>
        {
          // Init an RNG and the item IDs set once for partition
          val random = new Random()
          val allItemIDs = bAllItemIDs.value
          userIDAndPosItemIDs.map {
            case (userID, posItemIDs) =>
              val posItemIDSet = posItemIDs.toSet
              val negative = new ArrayBuffer[Int]()
              var i = 0
              // Keep about as many negative examples per user as positive.
              // Duplicates are OK
              while (i < allItemIDs.size && negative.size < posItemIDSet.size) {
                val itemID = allItemIDs(random.nextInt(allItemIDs.size))
                if (!posItemIDSet.contains(itemID)) {
                  negative += itemID
                }
                i += 1
              }
              // Result is a collection of (user,negative-item) tuples
              negative.map(itemID => (userID, itemID))
          }
        }
    }.flatMap(t => t)
    // flatMap breaks the collections above down into one big set of tuples

    // Make predictions on the rest:
    val negativePredictions = predictFunction(negativeUserProducts).groupBy(_.user)

    // Join positive and negative by user
    positivePredictions.join(negativePredictions).values.map {
      case (positiveRatings, negativeRatings) =>
        // AUC may be viewed as the probability that a random positive item scores
        // higher than a random negative one. Here the proportion of all positive-negative
        // pairs that are correctly ranked is computed. The result is equal to the AUC metric.
        var correct = 0L
        var total = 0L
        // For each pairing,
        for (
          positive <- positiveRatings;
          negative <- negativeRatings
        ) {
          // Count the correctly-ranked pairs
          if (positive.rating > negative.rating) {
            correct += 1
          }
          total += 1
        }
        // Return AUC: fraction of pairs ranked correctly
        correct.toDouble / total
    }.mean() // Return mean AUC over users
  }
  
  def unpersist(model: MatrixFactorizationModel): Unit = {
    // At the moment, it's necessary to manually unpersist the RDDs inside the model
    // when done with it in order to make sure they are promptly uncached
    model.userFeatures.unpersist()
    model.productFeatures.unpersist()
  }
    
}