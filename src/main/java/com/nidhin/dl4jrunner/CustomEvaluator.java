package com.nidhin.dl4jrunner;

import com.nidhin.urlclassifier.WordKmeans;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.*;

/**
 * Created by nidhin on 27/7/17.
 */
public class CustomEvaluator {
    HashMap<String, ArrayList<String>> aspirationUrlsMap = new HashMap<>();
//    HashMap<String, WordKmeans.Classes[]> urlClustersMap = new HashMap<>();
    HashMap<String, HashSet<WordKmeans.Classes>> urlTopClustersMap = new HashMap<String, HashSet<WordKmeans.Classes>>();
    MultiLayerNetwork model = null;

    public void loadMapsAndModel() throws IOException, ClassNotFoundException {
        ObjectInputStream /*ois = new ObjectInputStream(new FileInputStream("/home/nidhin/Jump2/jump-classifier/urlClustersMap-google.ser"));
        urlClustersMap = (HashMap<String, WordKmeans.Classes[]>) ois.readObject();
        ois.close();*/
        ois = new ObjectInputStream(new FileInputStream("/home/nidhin/Jump2/jump-classifier/aspirationUrlsMap.ser"));
        aspirationUrlsMap = (HashMap<String, ArrayList<String>>) ois.readObject();
        ois.close();
        ois = new ObjectInputStream(new FileInputStream("/home/nidhin/Jump2/jump-classifier/urlClustersMap-top10perpage-google.ser"));
        urlTopClustersMap = (HashMap<String, HashSet<WordKmeans.Classes>>) ois.readObject();
        ois.close();
        ois = new ObjectInputStream(new FileInputStream("/home/nidhin/Jump2/dl4jrunner/mlp-dl4j-google-top10-4-150.ser"));
        model = (MultiLayerNetwork) ois.readObject();
        ois.close();
    }

    public void loadMaps() throws IOException, ClassNotFoundException {
        ObjectInputStream /*ois = new ObjectInputStream(new FileInputStream("/home/ubuntu/url2category/urlClustersMap-google.ser"));
        urlClustersMap = (HashMap<String, WordKmeans.Classes[]>) ois.readObject();
        ois.close();*/
        ois = new ObjectInputStream(new FileInputStream("/home/ubuntu/url2category/aspirationUrlsMap.ser"));
        aspirationUrlsMap = (HashMap<String, ArrayList<String>>) ois.readObject();
        ois.close();
        ois = new ObjectInputStream(new FileInputStream("/home/ubuntu/url2category/urlClustersMap-top10perpage-google.ser"));
        urlTopClustersMap = (HashMap<String, HashSet<WordKmeans.Classes>>) ois.readObject();
        ois.close();

    }

    public void loadModel(String modelPath) throws IOException, ClassNotFoundException {
        ObjectInputStream  ois = new ObjectInputStream(new FileInputStream(modelPath));
        model= null;
        model = (MultiLayerNetwork) ois.readObject();
        ois.close();
    }

//    public void testModel(){

//        ModelEvaluator modelEvaluator = new ModelEvaluator(8);

//        HashMap<String, Integer> urlLabelsMap = new HashMap<>();
//        ArrayList<String> sortedLabelNames = new ArrayList<>(aspirationUrlsMap.keySet());
//        Collections.sort(sortedLabelNames);

//        for (Map.Entry<String, ArrayList<String>> aspUrls : aspirationUrlsMap.entrySet()){
//            String asp = aspUrls.getKey();
//            ArrayList<String> urls = aspUrls.getValue();
//            Integer label = sortedLabelNames.indexOf(asp);
//            for (String url : urls){
//                urlLabelsMap.put(url, label);
//            }
//        }
//        int tcorrectClusterPred =0, twrongClusterPreds =0, tcluster =0;
//        int[] correctasp = new int[8];
//        int[] wrongasp = new int[8];

//        for (Map.Entry<String, WordKmeans.Classes[]> urlClusters : urlClustersMap.entrySet()){

//            String url = urlClusters.getKey();
//            WordKmeans.Classes[] clusters = urlClusters.getValue();
//            Integer correctLabel = urlLabelsMap.get(url);

//            int correctClusterPred =0, wrongClusterPreds =0;
//            int predictions[] = new int[8];
//            for ( int i =0; i< clusters.length; i++){
//                if (String.valueOf(clusters[i].getCenter()[0]).equals("NaN")){
//                    continue;
//                }
//                tcluster++;
//                INDArray input = Nd4j.create(clusters[i].getCenter());
//                int pred = model.predict(input)[0];

//                predictions[pred]++;

//                if (pred == correctLabel){
//                    correctClusterPred++;
//                }
//                else {
//                    wrongClusterPreds++;
//                }
//            }

//            int finalPreLabel = getMaxPredIndex(predictions);
//            if (finalPreLabel == correctLabel){

//                tcorrectClusterPred++;
//                correctasp[correctLabel]++;
//            }
//            else {
//                twrongClusterPreds++;
//                wrongasp[correctLabel]++;
//            }
//            modelEvaluator.addRecord(finalPreLabel, correctLabel);

////            System.out.println(String.format("%s --- %d  : %d", url, correctLabel, finalPreLabel));
//        }
//        System.out.println("Tcluster - " + tcluster);
//        System.out.println(tcorrectClusterPred + "  :  " + twrongClusterPreds);

//        System.out.println("Accuracy - " + (tcorrectClusterPred *1.0)/((tcorrectClusterPred + twrongClusterPreds) * 1.0));
//        System.out.print("C - ");
//        for (int x = 0; x< correctasp.length; x++){
//            System.out.print(x+":" +correctasp[x] + ", ");
//        }
//        System.out.println();
//        System.out.print("W - ");
//        for (int x = 0; x< wrongasp.length; x++){
//            System.out.print(x+":"+wrongasp[x] + ", ");
//        }
//        System.out.println();
//        System.out.print("T - ");
//        int t=0;
//        for (int x = 0; x< wrongasp.length; x++){
//            int it = correctasp[x] + wrongasp[x];
//            t+= it;
//            System.out.print(x+":"+ it + ", ");
//        }
//        System.out.println();
//        System.out.println("t - " + t);

//        modelEvaluator.calcMetrics();
//        for (int i=0 ; i < 8; i++){

//            System.out.println(modelEvaluator.getClassScores(i));
//        }

//    }

    public void testModelWithTopCluster(){



        HashMap<String, Integer> urlLabelsMap = new HashMap<>();
        ArrayList<String> sortedLabelNames = new ArrayList<>(aspirationUrlsMap.keySet());
        Collections.sort(sortedLabelNames);

        for (Map.Entry<String, ArrayList<String>> aspUrls : aspirationUrlsMap.entrySet()){
            String asp = aspUrls.getKey();
            ArrayList<String> urls = aspUrls.getValue();
            Integer label = sortedLabelNames.indexOf(asp);
            for (String url : urls){
                urlLabelsMap.put(url, label);
            }
        }
        int tcorrectClusterPred =0, twrongClusterPreds =0, tcluster =0;
        int[] correctasp = new int[8];
        int[] wrongasp = new int[8];

        ModelEvaluator modelEvaluator = new ModelEvaluator(8);

        for (Map.Entry<String, HashSet<WordKmeans.Classes>> urlClusters : urlTopClustersMap.entrySet()){

            String url = urlClusters.getKey();
            HashSet<WordKmeans.Classes> clusters = urlClusters.getValue();
            Integer correctLabel = urlLabelsMap.get(url);

            int correctClusterPred =0, wrongClusterPreds =0;
            int predictions[] = new int[8];
            for ( WordKmeans.Classes cluster : clusters){
                if (String.valueOf(cluster.getCenter()[0]).equals("NaN")){
                    continue;
                }
                tcluster++;
                INDArray input = Nd4j.create(cluster.getCenter());
                int pred = model.predict(input)[0];

                predictions[pred]++;

                if (pred == correctLabel){
                    correctClusterPred++;
                }
                else {
                    wrongClusterPreds++;
                }
            }

            int finalPreLabel = getMaxPredIndex(predictions);
            if (finalPreLabel == correctLabel){
                tcorrectClusterPred++;
                correctasp[correctLabel]++;
            }
            else {
                twrongClusterPreds++;
                wrongasp[correctLabel]++;
            }
            modelEvaluator.addRecord(finalPreLabel, correctLabel);

//            System.out.println(String.format("%s --- %d  : %d", url, correctLabel, finalPreLabel));
        }
        System.out.println("tcluster - "+ tcluster);
        System.out.println(tcorrectClusterPred + "  :  " + twrongClusterPreds);
        System.out.println("Accuracy - " + (tcorrectClusterPred *1.0)/((tcorrectClusterPred + twrongClusterPreds) * 1.0));
        System.out.print("C - ");
        for (int x = 0; x< correctasp.length; x++){
            System.out.print(x+":" +correctasp[x] + ", ");
        }
        System.out.println();
        System.out.print("W - ");
        for (int x = 0; x< wrongasp.length; x++){
            System.out.print(x+":"+wrongasp[x] + ", ");
        }
        System.out.println();
        System.out.print("T - ");
        int t =0;
        for (int x = 0; x< wrongasp.length; x++){
            int it = correctasp[x] + wrongasp[x];
            t+= it;
            System.out.print(x+":"+ it + ", ");
        }
        System.out.println();
        System.out.println("t - " + t);

        modelEvaluator.calcMetrics();
        for (int i=0 ; i < 8; i++){

            System.out.println(modelEvaluator.getClassScores(i));
        }
    }

    public int getMaxPredIndex(int[] predictions){
        int predLabel = 0;
        int max = predictions[0];
        for (int i = 1; i< predictions.length; i++){
            if (predictions[i] > max){
                max = predictions[i];
                predLabel = i;
            }
        }
        return predLabel;
    }

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        CustomEvaluator customEvaluator = new CustomEvaluator();
        customEvaluator.loadMapsAndModel();
//        System.out.println("Test With All clusters");
//        customEvaluator.testModel();
        System.out.println("Test With Top clusters");
        customEvaluator.testModelWithTopCluster();

    }
}
