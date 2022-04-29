package com.rah.cs643.wqualityp;


import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import scala.Tuple2;
/***
 * CS643 Programming Assignment #2: Wine Quality Prediction 
 * @author Roderic Henry
 *
 * Main application entry class.
 */
@SpringBootApplication
public class WqualitypApplication implements CommandLineRunner {

	protected Logger log = LoggerFactory.getLogger(getClass());
	
	@Value("${dataset.training.path:#{null}}")
	private String trainingDataset;
	
	@Value("${dataset.test.path:#{null}}")
	private String testDataset;
	
	public static void main(String[] args) {
		SpringApplication.run(WqualitypApplication.class, args);
	}

	/**
	 * Spring boot's command line entry point. This is similar to main.
	 */
	@Override
    public void run(String... args) throws Exception {
		// If the test dataset is provided override the configured value.
        if (args.length > 0) {
    		testDataset = args[0];			
		}
        
        log.info("Processing file: {}", testDataset);
        if (testDataset == null) {
			log.error("Please provide a valid path to the test dataset. Usage: $> java -jar cs643-pa2-roderic-henry.jar <PATH TO TEST DATASET>");
			return;
		}
        
        // Set the Spark context.
		SparkConf conf = new SparkConf().setAppName("CS643PA2RodericHenry").setMaster("local[4]");
		JavaSparkContext sc = new JavaSparkContext(conf);
		
		// Load the training dataset.
		JavaRDD<String> trainingRDD = loadDataset(sc, trainingDataset);
		
		// Load the test dataset.
		JavaRDD<String> testRDD = loadDataset(sc, testDataset);
				
		// Create labeled point objects for training.
		JavaRDD<LabeledPoint> trainingData = createLabelPoint(trainingRDD);
		JavaRDD<LabeledPoint> testData = createLabelPoint(testRDD);
		
		// Create and train the model.
		LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
				.setNumClasses(10)
				.run(trainingData.rdd());
		JavaPairRDD<Object, Object> predictionAndLabels = testData.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
		MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
		double f1Score = metrics.weightedRecall();
		//log.info("***** The model's F1 score is {}", f1Score);
		
		// Close the spark context to release resources.
		sc.close();
				
		log.info("Complete. The F1 or recall score is {}", f1Score);
	}
	
	private JavaRDD<LabeledPoint> createLabelPoint(JavaRDD<String> rdd){
		JavaRDD<LabeledPoint> labeledData = rdd.map(line -> {
		      String[] parts = line.split(";");
		      double[] v = new double[parts.length - 1];
		      for (int i = 0; i < parts.length - 1; i++) {
		          v[i] = Double.parseDouble(parts[i]);
		      }
		      return new LabeledPoint(Integer.parseInt(parts[parts.length - 1]), Vectors.dense(v));
		});
		
		return labeledData;
	}
	
	private JavaRDD<String> loadDataset(JavaSparkContext sc, String path){
		JavaRDD<String> dataWithHeader = sc.textFile(path);
		final String header = dataWithHeader.first();
		JavaRDD<String> rdd = dataWithHeader.filter(line -> {
            return !line.equalsIgnoreCase(header);
        });
		
		return rdd;
	}
}
