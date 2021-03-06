package com.nidhin.dl4jrunner;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.play.PlayUIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;

/**
 * Created by nidhin on 27/7/17.
 */
public class WordVecClassifier {
    private int batchSize = 512, labelIndex = 0, numClasses = 8;
    private DataSetIterator iterator, evalIterator;
    private double learningRate = 0.05;
    private int nEpochs = 400;
    static Logger logger = LoggerFactory.getLogger(WordVecClassifier.class);

    public WordVecClassifier() {

    }

    public void init() throws IOException, InterruptedException {
        RecordReader trainrr = new CSVRecordReader();
        trainrr.initialize(new FileSplit(new File("mlp-large-top10-google-train.csv")));
        iterator = new RecordReaderDataSetIterator(trainrr, batchSize, labelIndex, numClasses);
        RecordReader evalrr = new CSVRecordReader();
        evalrr.initialize(new FileSplit(new File("mlp-large-top10-google-eval.csv")));
        evalIterator = new RecordReaderDataSetIterator(evalrr, batchSize, labelIndex, numClasses);
    }

    public void process( int port) throws IOException, ClassNotFoundException {
//        System.setProperty("org.deeplearning4j.ui.port", "5000");


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345l)
                .iterations(1)
                .activation(Activation.LEAKYRELU)
                .weightInit(WeightInit.XAVIER)
//                .activation(Activation.RELU)
//                .weightInit(WeightInit.XAVIER)
                .updater(Updater.RMSPROP)
//                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
//                .gradientNormalizationThreshold(0.9)
//                .updater(Updater.ADAGRAD)
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.3)
                .learningRateDecayPolicy(LearningRatePolicy.Score)
                .lrPolicyDecayRate(0.7)
                .regularization(true)
//                .l2(1e-6)
                .dropOut(0.5)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(300).nOut(500).build())
                .layer(1, new DenseLayer.Builder().nIn(500).nOut(700).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX).nIn(700).nOut(numClasses).learningRate(0.1)
                        .build())
                .backprop(true).pretrain(false)
                .build();
//
//
//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//            .seed(12345)
//            .iterations(1)
//            .activation(Activation.RELU)
//            .weightInit(WeightInit.RELU)
//            .learningRate(0.1)
//            .updater(Updater.ADAM)
//            .regularization(true).l2(1e-4)
//            .list()
//            .layer(0, new DenseLayer.Builder().nIn(100).nOut(150).build())
//            .layer(1, new DenseLayer.Builder().nIn(200).nOut(250).build())
//            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                .activation(Activation.SOFTMAX).nIn(250).nOut(numClasses).build())
//            .pretrain(false).backprop(true)
//            .build();


        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        PlayUIServer playUIServer = new PlayUIServer();
        playUIServer.runMain(new String[]{"--uiPort", String.valueOf(port)});
        UIServer uiServer = playUIServer;


        StatsStorage statsStorage = new InMemoryStatsStorage();
        int listenerFrequency = 10;
        ArrayList<IterationListener> listeners = new ArrayList<>();
        listeners.add(new StatsListener(statsStorage, listenerFrequency));
//        listeners.add(new ScoreIterationListener(50));
        model.setListeners(listeners);
        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);

//        model.setListeners(Arrays.asList(,new ScoreIterationListener(10)));

        CustomEvaluator customEvaluator = new CustomEvaluator();
        customEvaluator.loadMaps();


        for (int i = 0; i < nEpochs; i++) {
            iterator.reset();
            while (iterator.hasNext()) {
                DataSet ds = iterator.next();
                //ds.normalize();
                model.fit(ds);
            }
            logger.info("epoch - " + i + " over.");

            if (i != 0 && i% 5 ==0){
                Evaluation evaluation = new Evaluation(numClasses);
                evalIterator.reset();
                while (evalIterator.hasNext()){
                    DataSet ds = evalIterator.next();
                    //ds.normalize();
                    evaluation.eval(ds.getLabels(), model.output(ds.getFeatures()));
                }

                logger.info("Acc - " + evaluation.accuracy() + " F1 - " + evaluation.f1() +
                " Pre - " + evaluation.precision() + " Rec - " + evaluation.recall());
            }
            if ((i != 0 && i%10 == 0)  || i==5){
                model.setListeners(new ArrayList<>());
                ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(String.format("mlp-dl4j-google-top10-4-%d.ser", i)));
                oos.writeObject(model);
                oos.flush();
                oos.close();
                model.setListeners(listeners);


                customEvaluator.loadModel(String.format("mlp-dl4j-google-top10-4-%d.ser", i));
//                logger.info("Test With All clusters");
//                customEvaluator.testModel();
                logger.info("Test With Top clusters");
                customEvaluator.testModelWithTopCluster();
            }
        }

        //evaluate the model on the test set
        Evaluation eval = new Evaluation(numClasses);

        iterator.reset();
        while (iterator.hasNext()) {
            DataSet ds = iterator.next();
            //ds.normalize();
            eval.eval(ds.getLabels(), model.output(ds.getFeatures()));
        }
        logger.info(eval.stats());
        model.setListeners(new ArrayList<>());


        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(String.format("mlp-dl4j-google-top10-4-%d.ser", nEpochs)));
        oos.writeObject(model);
        oos.flush();
        oos.close();
        logger.info("model saved");


    }

//    public MultiLayerConfiguration getNNConf(){
//        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(12345l)
//                .iterations(1)
//                .activation(Activation.LEAKYRELU)
//                .weightInit(WeightInit.XAVIER)
////                .activation(Activation.RELU)
////                .weightInit(WeightInit.XAVIER)
//                .updater(Updater.NESTEROVS).momentum(0.9)
////                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
////                .gradientNormalizationThreshold(0.9)
////                .updater(Updater.ADAGRAD)
//                .learningRate(0.1)
//                .learningRateDecayPolicy(LearningRatePolicy.Exponential)
//                .regularization(true).l2(1e-4)
//                .list()
//                .layer(0, new DenseLayer.Builder().nIn(300).nOut(500).build())
//                .layer(1, new DenseLayer.Builder().nIn(500).nOut(700).build())
//                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .activation(Activation.SOFTMAX).nIn(700).nOut(numClasses).learningRate(0.01)
//                        .build())
//                .backprop(true).pretrain(false)
//                .build();
//        return conf;
//    }

//    public void earlyStopProcess(){
//        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
//                .epochTerminationConditions(new MaxEpochsTerminationCondition(100))
//                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES))
//                .scoreCalculator(new DataSetLossCalculator(evalIterator, true))
//                .evaluateEveryNEpochs(1)
//                .modelSaver(new LocalFileModelSaver("es_models"))
//                .build();

//        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,getNNConf(),iterator);

////Conduct early stopping training:
////        UIServer uiServer = UIServer.getInstance();


////        StatsStorage statsStorage = new InMemoryStatsStorage();
////        trainer.setListener(new Ea);
//        EarlyStoppingResult result = trainer.fit();

////Print out the results:
//        logger.info("Termination reason: " + result.getTerminationReason());
//        logger.info("Termination details: " + result.getTerminationDetails());
//        logger.info("Total epochs: " + result.getTotalEpochs());
//        logger.info("Best epoch number: " + result.getBestModelEpoch());
//        logger.info("Score at best epoch: " + result.getBestModelScore());

////Get the best model:
//        Model bestModel = result.getBestModel();

//    }

    public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException {
        WordVecClassifier wordVecClassifier = new WordVecClassifier();
        wordVecClassifier.init();
        wordVecClassifier.process(Integer.parseInt(args[0]));
//        wordVecClassifier.earlyStopProcess();
    }
}
