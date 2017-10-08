package com.nidhin.dl4jrunner;

public class ModelEvaluator {

    private int[] truePositives, falsePositives, trueNegatives, falseNegatives;
    private double[] accuracy, precision, recall, f1score;
    private int totalFalsePositives, totalFalseNegatives, totalTruePositives, totalTrueNegatives, numClasses;
    private int defaultVal = 0;


    public ModelEvaluator(int numClasses){
        this.numClasses = numClasses;
        this.truePositives = new int[numClasses];
        this.falsePositives = new int[numClasses]; // predicted as class a, but is actually b, false positive of a
        this.trueNegatives = new int[numClasses];
        this.falseNegatives = new int[numClasses]; // predicted as class a, but is actually b, false negative of b
        this.accuracy = new double[numClasses];
        this.precision = new double[numClasses];
        this.recall = new double[numClasses];
        this.f1score = new double[numClasses];

    }


    public void addRecord(int predictedLabel, int correctLabel){

        if (predictedLabel == correctLabel){
            truePositives[correctLabel] ++;
            for (int i = 0; i< numClasses; i++){
              if (i != correctLabel)
                  trueNegatives[i] ++;
            }
            totalTruePositives ++;
            totalTrueNegatives += (numClasses - 1);
        }
        else {
            falsePositives[predictedLabel] ++;
            falseNegatives[correctLabel] ++;
            for (int i = 0; i< numClasses; i++){
                if ((i != predictedLabel) && (i != correctLabel))
                    trueNegatives[i] ++;
            }
            totalFalsePositives ++;
            totalFalseNegatives ++;
            totalTrueNegatives += (numClasses - 2);
        }
    }

    public void calcMetrics(){

        for (int i = 0; i< numClasses; i++){
            if (truePositives[i] + falsePositives[i]  != 0)
                precision[i] = (truePositives[i] * 1.0)/(truePositives[i] + falsePositives[i]);
            else
                precision[i] = defaultVal;
            if (truePositives[i] + falseNegatives[i] != 0)
                recall[i] = (truePositives[i] * 1.0)/(truePositives[i] + falseNegatives[i]);
            else
                recall[i] = defaultVal;

            f1score[i] = (2 * precision[i] * recall[i])/(precision[i] + recall[i]);
        }
    }

    public double[] getPrecision() {
        return precision;
    }

    public void setPrecision(double[] precision) {
        this.precision = precision;
    }

    public double[] getRecall() {
        return recall;
    }

    public void setRecall(double[] recall) {
        this.recall = recall;
    }

    public double[] getF1score() {
        return f1score;
    }

    public void setF1score(double[] f1score) {
        this.f1score = f1score;
    }

    public int getTotalFalsePositives() {
        return totalFalsePositives;
    }

    public void setTotalFalsePositives(int totalFalsePositives) {
        this.totalFalsePositives = totalFalsePositives;
    }

    public int getTotalFalseNegatives() {
        return totalFalseNegatives;
    }

    public void setTotalFalseNegatives(int totalFalseNegatives) {
        this.totalFalseNegatives = totalFalseNegatives;
    }

    public int getTotalTruePositives() {
        return totalTruePositives;
    }

    public void setTotalTruePositives(int totalTruePositives) {
        this.totalTruePositives = totalTruePositives;
    }

    public int getTotalTrueNegatives() {
        return totalTrueNegatives;
    }

    public void setTotalTrueNegatives(int totalTrueNegatives) {
        this.totalTrueNegatives = totalTrueNegatives;
    }

    public String getClassScores(int i){
        String str = String.format(" Label - %d, TP - %d, TN - %d, FP - %d, FN - %d, P - %f, R - %f, F1 - %f", i,
                truePositives[i], trueNegatives[i], falsePositives[i], falseNegatives[i],
                precision[i], recall[i], f1score[i]);
        return str;
    }
}
