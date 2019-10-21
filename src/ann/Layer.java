package ann;

import java.util.Random;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

/**
 *
 * @author Tasnim
 */
public class Layer {

    RealMatrix Z, W, D, S, Fp;
    // W is the outgoing weight matrix for this layer
    // Z is the matrix that holds output values  
    // S is the matrix that holds the inputs to this layer
    // D is the matrix that holds the deltas for this layer
    // Fp is the matrix that holds the derivatives of the activation function
    boolean input;
    boolean output;
    Integer numNodesThisLayer;
    Integer numNodesNextLayer;
    int minibatchSize;
    
    public Layer(Integer numNodesThisLayer, Integer numNodesNextLayer, boolean input, boolean output, int minibatchSize) {
        this.input = input;
        this.output = output;
        this.numNodesThisLayer = numNodesThisLayer;
        this.numNodesNextLayer = numNodesNextLayer;
        this.minibatchSize = minibatchSize;

        Z = MatrixUtils.createRealMatrix(minibatchSize, numNodesThisLayer);
        W = null;
        S = null;
        D = null;
        Fp = null;

        Random random = new Random(0);
        if (!input) {
            S = MatrixUtils.createRealMatrix(minibatchSize, numNodesThisLayer);
            D = MatrixUtils.createRealMatrix(minibatchSize, numNodesThisLayer);
        }
        if (!output) {
            W = MatrixUtils.createRealMatrix(numNodesThisLayer, numNodesNextLayer);
            ///*
            for (int i = 0; i < W.getRowDimension(); i++) {
                for (int j = 0; j < W.getColumnDimension(); j++) {
                    //W.setEntry(i, j, getGaussian(random, 0, 1, 0.1));
                    W.setEntry(i, j, getGaussian(random, 0, 1, 1/Math.sqrt(numNodesThisLayer)));
                }
            }
    
        }

        if (input == false && output == false) {
            Fp = MatrixUtils.createRealMatrix(numNodesThisLayer, minibatchSize);
        }

    }

    private double getGaussian(Random random, double aMean, double aVariance, double aScale) {
        return (aMean + random.nextGaussian() * aVariance) * aScale;
    }

    RealMatrix forwardPropagate() {
        if (input == true) {
            return Z.multiply(W);
        }
        Z = applyActivation(S, false);
        if (output == true) {
            Fp = applyActivation(Z, true).transpose();
            return Z;
        } else {
            Fp = applyActivation(Z, true).transpose();
            //hidden layer
            //add bias, make a new column at the end, Z = numexamples * numnodesThisLayer
            double[][] newZ = new double[Z.getRowDimension()][Z.getColumnDimension() + 1];
            for (int i = 0; i < Z.getRowDimension(); i++) {
                for (int j = 0; j <  Z.getColumnDimension(); j++) {
                    newZ[i][j] = Z.getEntry(i, j);
                }    
            }
            for (int i = 0; i < Z.getRowDimension(); i++) {
                newZ[i][Z.getColumnDimension()] = 1;
            }
            Z = MatrixUtils.createRealMatrix(newZ);
            //Fp = applyActivation(S, true).transpose();
            return Z.multiply(W);
        }
    }

    public RealMatrix applyActivation(RealMatrix X, boolean deriv) {
        
        double[][] mat = new double[X.getRowDimension()][X.getColumnDimension()];
        
        /*if (deriv == false) {
            double max = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < X.getRowDimension(); i++) {
                for (int j = 0; j < X.getColumnDimension(); j++) {
                    if (X.getEntry(i, j) > max) {
                        max = X.getEntry(i, j);
                    }
                }
            }
            for (int i = 0; i < X.getRowDimension(); i++) {
                double sameRowSum = 0.0;
                for (int j = 0; j < X.getColumnDimension(); j++) {
                   sameRowSum += Math.exp(X.getEntry(i, j) - max);
                }
                for (int j = 0; j < X.getColumnDimension(); j++) {
                    mat[i][j] = Math.exp(X.getEntry(i, j) - max)/sameRowSum;
                }
            }
        } else {
            for (int i = 0; i < X.getRowDimension(); i++) {
                for (int j = 0; j < X.getColumnDimension(); j++) {
                    mat[i][j] = X.getEntry(i, j) * (1 - X.getEntry(i, j));
                }
            }
        }*/
        
        if(!this.output){
            for (int i = 0; i < X.getRowDimension(); i++) {
                for (int j = 0; j < X.getColumnDimension(); j++) {
                    mat[i][j] = f_sigmoid(X.getEntry(i, j), deriv);
                }
            }
        }
        //softmax
        if (this.output){
            // eki row er column dhore jog kore totogula row answer
            // age joto row chilo totoi thakbe, column hobe ekta
            // Z = ekta row er shob element er sum
            // row er shob col element ke vaag dite hobe column sum diye
            for (int i = 0; i < X.getRowDimension(); i++) {
                double max = Double.NEGATIVE_INFINITY;
                for (int j = 0; j < X.getColumnDimension(); j++) {
                   if (X.getEntry(i, j) > max) {
                       max = X.getEntry(i, j);
                   }
                }
                double sameRowSum = 0.0;
                for (int j = 0; j < X.getColumnDimension(); j++) {
                   sameRowSum += Math.exp(X.getEntry(i, j)-max);
                }
                for (int j = 0; j < X.getColumnDimension(); j++) {
                    mat[i][j] = Math.exp(X.getEntry(i, j)-max)/sameRowSum;
                }
                for (int j = 0; j < X.getColumnDimension(); j++) {
                    if (deriv == true) {
                        mat[i][j] = X.getEntry(i, j) * (1 - X.getEntry(i, j));
                    }
                }
            }   
        }
        
        return MatrixUtils.createRealMatrix(mat);
        
    }

    public double f_sigmoid(double X, boolean deriv) {
        if (deriv == false) {
            return 1 / (1 + Math.exp(-X));
        } else {
            //return f_sigmoid(X, false) * (1 - f_sigmoid(X, false));
            return X * (1 - X);
        }
    }
    
  
}
