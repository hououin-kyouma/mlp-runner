package com.nidhin.dl4jrunner;

import com.nidhin.urlclassifier.WordKmeans;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by nidhin on 16/8/17.
 */
public class DistBasedClassifier {
    public String predictUrlTypeBasedOnClusterCloseness(HashMap<String, HashSet<double[]>> aspirationClusters, HashSet<double[]> clusters){

        HashMap<String, HashMap<String, Object>> aspirationScores = new HashMap<>();
        for (double[] urlCluster : clusters){



            String selectedAsp = null;
            double selectedAspDist = Double.MAX_VALUE;
            for (String aspiration : aspirationClusters.keySet()){
                double aspMinDist = Double.MAX_VALUE;
                for (double[] aspCluster : aspirationClusters.get(aspiration)){
                    double dist = distance(urlCluster, aspCluster);
                    if (dist < aspMinDist)
                        aspMinDist = dist;
                }
                if (aspMinDist < selectedAspDist){
                    selectedAspDist = aspMinDist;
                    selectedAsp = aspiration;
                }
            }
            if (!aspirationScores.containsKey(selectedAsp)) {
                HashMap<String, Object> aspScore = new HashMap<>();
                aspScore.put("total", 0.0);
                aspScore.put("count", 0);
                aspScore.put("vals", new ArrayList<Double>());
                aspirationScores.put(selectedAsp, aspScore);
            }

            aspirationScores.get(selectedAsp).put("total", selectedAspDist + (Double) aspirationScores.get(selectedAsp).get("total"));
            aspirationScores.get(selectedAsp).put("count", 1 + (Integer) aspirationScores.get(selectedAsp).get("count"));
            ((ArrayList<Double>)aspirationScores.get(selectedAsp).get("vals")).add(selectedAspDist);
        }

        List<Map.Entry<String, HashMap<String, Object>>> sortedAsps = aspirationScores.entrySet()
                .stream()
                .map(stringHashMapEntry -> {
                    Double total = (Double) stringHashMapEntry.getValue().get("total");
                    Integer count = (Integer) stringHashMapEntry.getValue().get("count");
                    double avg = total/count;
                    stringHashMapEntry.getValue().put("avg", avg);
                    return stringHashMapEntry;
                })
                .sorted((o1, o2) -> {
                    Double avg1 = (Double) o1.getValue().get("avg");
                    Double avg2 = (Double) o2.getValue().get("avg");
                    return avg1.compareTo(avg2);
                })
                .collect(Collectors.toList());

//       String mostSelectedAsp = null;
//        int mostDetectedAspCount =0, totaldetectedAspCount = 0;
//       for (String asp : detectedAspiration.keySet()){
//           int detectedCount = detectedAspiration.get(asp);
//           if (detectedCount > mostDetectedAspCount){
//               mostDetectedAspCount = detectedCount;
//               mostSelectedAsp = asp;
//           }
//           totaldetectedAspCount += detectedCount;
//       }

        HashMap<String, Object> rslts = new HashMap<>();
//        rslts.put("class", mostSelectedAsp);
//        rslts.put("score", (mostDetectedAspCount/ (totaldetectedAspCount * 1.0)));
//        if (sortedAsps.isEmpty()){
//            rslts.put("class", "Not defined");
//            rslts.put("score", -88.0);
//        }else {
//            rslts.put("class", sortedAsps.get(0).getKey());
//            rslts.put("score", sortedAsps.get(0).getValue().get("avg"));
//        }
        if (sortedAsps.isEmpty()){
            return null;
        }
        return sortedAsps.get(0).getKey();
    }

    public double distance(double[] vector1, double[] vector2) {

        double sumOfSquares = 0;
        for (int i = 0; i< vector1.length; i++){
            sumOfSquares += Math.pow(vector1[i] - vector2[i], 2);
        }
        return Math.sqrt(sumOfSquares);
    }

    public static void main(String[] args) throws IOException, ClassNotFoundException {

        DistBasedClassifier distBasedClassifier = new DistBasedClassifier();

        ObjectInputStream ois = new ObjectInputStream(new FileInputStream("/home/nidhin/Jump2/jump-classifier/aspirationClustersMap-top20perpage-google.ser"));
        HashMap<String, HashSet<double[]>> aspirationClusters = (HashMap<String, HashSet<double[]>>) ois.readObject();
        ois.close();

        ois = new ObjectInputStream(new FileInputStream("/home/nidhin/Jump2/jump-classifier/urlClustersMap-top20perpage-google.ser"));
        HashMap<String, HashSet<double[]>> urlTopClustersMap = (HashMap<String, HashSet<double[]>>) ois.readObject();
        ois.close();

        ois = new ObjectInputStream(new FileInputStream("/home/nidhin/Jump2/jump-classifier/urlClustersMap-google.ser"));
        HashMap<String, WordKmeans.Classes[]> urlClustersMap = (HashMap<String, WordKmeans.Classes[]>) ois.readObject();
        ois.close();

        ois = new ObjectInputStream(new FileInputStream("/home/nidhin/Jump2/jump-classifier/aspirationUrlsMap.ser"));
        HashMap<String, ArrayList<String>> aspirationUrlsMap = (HashMap<String, ArrayList<String>>) ois.readObject();
        ois.close();

        ModelEvaluator modelEvaluator = new ModelEvaluator(8);

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
        int urlCount = 0, nullcount = 0, totalcount = urlClustersMap.keySet().size();

        System.out.println("All cluster");

        ModelEvaluator modelEvaluator1 = new ModelEvaluator(8);

        for (String url : urlClustersMap.keySet()){

            HashSet<double[]> urlClusters = new HashSet<>();
            WordKmeans.Classes[] clusters = urlClustersMap.get(url);
            for (int x=0; x< clusters.length; x++){

                urlClusters.add(clusters[x].getCenter());
            }

            String asp =  distBasedClassifier.predictUrlTypeBasedOnClusterCloseness(aspirationClusters, urlClusters);
            int correctLabel = urlLabelsMap.get(url);
            urlCount++;
            if (asp == null) {
                nullcount ++;
                continue;
            }
            int predLabel = sortedLabelNames.indexOf(asp);

            modelEvaluator1.addRecord(predLabel, correctLabel);
            if (urlCount %50 == 0){
//              Runtime.getRuntime().exec("clear");
                System.out.println("Total - " + totalcount + " done - " + urlCount + " null - " + nullcount);
            }
        }

        modelEvaluator1.calcMetrics();
        for (int i = 0; i<8; i++){

            System.out.println(modelEvaluator1.getClassScores(i));
        }

        System.out.println("Top clusters - ");
        urlCount = 0; nullcount = 0; totalcount = urlTopClustersMap.keySet().size();



        for (String url : urlTopClustersMap.keySet()){

          String asp =  distBasedClassifier.predictUrlTypeBasedOnClusterCloseness(aspirationClusters, urlTopClustersMap.get(url));
          int correctLabel = urlLabelsMap.get(url);
          urlCount++;
          if (asp == null) {
              nullcount ++;
              continue;
          }
          int predLabel = sortedLabelNames.indexOf(asp);

          modelEvaluator.addRecord(predLabel, correctLabel);
          if (urlCount %50 == 0){
//              Runtime.getRuntime().exec("clear");
              System.out.println("Total - " + totalcount + " done - " + urlCount + " null - " + nullcount);
          }
        }



        modelEvaluator.calcMetrics();
        for (int i = 0; i<8; i++){

            System.out.println(modelEvaluator.getClassScores(i));
        }






    }
}
