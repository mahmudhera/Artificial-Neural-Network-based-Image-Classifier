package ann;

import static ann.ANN.MNISTDatasetTest;
import java.util.ArrayList;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class Network {

    Layer[] layers;
    int minibatchsize;
    ArrayList<Example> list;
    int numClasses;
    int numFeatures;
    ArrayList<Example>testSet;

    public Network(ArrayList<Example> list, int[] layers, int minibatchsize, ArrayList<Example> test) {
        
        testSet = test;
        this.numFeatures = layers[0];
        this.numClasses = layers[layers.length - 1];
        this.minibatchsize = minibatchsize;
        this.list = list;
        this.layers = new Layer[layers.length];

        /**
         * *** input layer at (0), then hidden layers, output layer at (numLayers-1)***
         */
        for (int i = 0; i < layers.length; i++) {
            // input layer
            if (i == 0) {
                System.out.println("Initializing input layer with size " + layers[i]);
                //add bias
                this.layers[0] = new Layer(layers[0] + 1, layers[1], true, false, minibatchsize);
            } 
            // output layer
            else if (i == layers.length - 1) {
                System.out.println("Initializing output layer with size " + layers[i]);
                //no bias is added since no layer follows
                this.layers[i] = new Layer(layers[i], null, false, true, minibatchsize);
                // softmax in this layer
            } else {
                System.out.printf("Initializing hidden layer %d layer with size %d \n", i, layers[i]);
                //add bias
                this.layers[i] = new Layer(layers[i] + 1, layers[i + 1], false, false, minibatchsize);
            }
        }

        System.out.printf("Network creation done!\n");

    }

    RealMatrix forwardPropagate(ArrayList<Example> list) {
        //number of examples, n
        int n = list.size();

        //number of features + 1 for bias, m
        int m = this.numFeatures + 1;

        double[][] nmMat = new double[n][m];
        for (int i = 0; i < n; i++) {
            Example ex = list.get(i);
            for (int j = 0; j < m - 1; j++) {
                nmMat[i][j] = ex.features.getEntry(j);
            }
            //add bias
            for (int j = m - 1; j < m; j++) {
                nmMat[i][j] = 1;
            }
        }
        //X => n * m matrix -> example vs. features
        RealMatrix X = MatrixUtils.createRealMatrix(nmMat);
        this.layers[0].Z = X;   //input layer

        for (int i = 0; i < layers.length - 1; i++) {
            this.layers[i + 1].S = this.layers[i].forwardPropagate();
        }
        return this.layers[layers.length - 1].forwardPropagate();

    }

    public void backPropagate(RealMatrix yhat, RealMatrix labelMatrix) {
        //row by row classes, column by column examples
        this.layers[layers.length - 1].D = yhat.subtract(labelMatrix).transpose(); 
        
        for (int j = 0; j < this.layers[layers.length-1].D.getColumnDimension(); j++) {
            RealVector col1 = this.layers[layers.length-1].D.getColumnVector(j);
            RealVector col2 = this.layers[layers.length-1].Fp.getColumnVector(j);
            this.layers[layers.length-1].D.setColumnVector(j, col1.ebeMultiply(col2));
        }
        
        //start from last hidden layer upto 1st hidden layer
        for (int i = layers.length - 2; i > 0; i--) {  
            //We do not calculate deltas for the bias values
            RealMatrix W_nobias = this.layers[i].W.getSubMatrix(0, this.layers[i].W.getRowDimension() - 2, 0, this.layers[i].W.getColumnDimension() - 1);
            this.layers[i].D = W_nobias.multiply(this.layers[i + 1].D);
          
            //column traversing => examples one by one
            for (int j = 0; j < this.layers[i].D.getColumnDimension(); j++) {
                RealVector col1 = this.layers[i].D.getColumnVector(j);
                RealVector col2 = this.layers[i].Fp.getColumnVector(j);
                this.layers[i].D.setColumnVector(j, col1.ebeMultiply(col2));
            }
        }
    }

    public void updateWeights(double eta) {
        //except output layer
        for (int i = 0; i < this.layers.length - 1; i++) {
            RealMatrix temp = this.layers[i + 1].D.multiply(this.layers[i].Z);
            RealMatrix W_grad = temp.transpose().scalarMultiply(-eta);
            this.layers[i].W = this.layers[i].W.add(W_grad);
        }
    }

    public void train(int numEpochs, double eta) {

        RealMatrix output = null;
        double prevLMSError = 1.0;

        System.out.printf("Training for %d epochs... \n", numEpochs);
        for (int t = 0; t < numEpochs; t++) {
 
            for (int i = 0; i < list.size(); i = i + minibatchsize) {

                ArrayList<Example> sublist = new ArrayList<>();
                RealMatrix labelMatrix = MatrixUtils.createRealMatrix(minibatchsize, numClasses);

                for (int j = 0; j < minibatchsize; j++) {
                    labelMatrix.setRow(j, list.get(i + j).outputs.toArray());
                    sublist.add(list.get(i + j));
                }
            
                output = this.forwardPropagate(sublist); //yhat
//                
//                double error = 0.0;
//                RealMatrix errors = output.subtract(labelMatrix);
//                for (int j = 0; j < errors.getRowDimension(); j++) {
//                    for (int k = 0; k < errors.getColumnDimension(); k++) {
//                        error += errors.getEntry(j, k)*errors.getEntry(j, k);
//                    }
//                }
//                if (t == 0) {
//                    prevLMSError = error;
//                } else {
//                    if ( Math.abs(prevLMSError-error)/Math.abs(prevLMSError+error) < 0.000005 ) {
//                        return;
//                    }
//                    prevLMSError = error;
//                }
                
                this.backPropagate(output, labelMatrix);
                this.updateWeights(eta);
                
                 
                    double accuracy = check(testSet);

                    System.out.println(accuracy);
                
            }
            
        }
    }
    
    public double check(ArrayList<Example> list) {
       
        int correct = 0;
        int incorrect = 0;
        
        for (int i = 0; i < list.size(); i++) {
            ArrayList<Example> sublist = new ArrayList<>();
            sublist.add(list.get(i));
            RealMatrix subOutput = this.forwardPropagate(sublist);
            int actualLabel = sublist.get(0).getLabel();
            int estimatedLabel = Example.getLabel(subOutput.getRowVector(0));
            //System.out.printf("Act %d Est %d\n", actualLabel, estimatedLabel);
            if (actualLabel == estimatedLabel) {
                correct++;
            } else {
                incorrect++;
            }

        }

        return 100.0 * correct / (correct + incorrect);
       
    }
}
