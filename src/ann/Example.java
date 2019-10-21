package ann;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Hera
 */
public class Example {

    RealVector features;
    RealVector outputs;

    public Example(RealVector features, RealVector outputs) {
        this.features = features;
        this.outputs = outputs;
    }

    public Example(double[] features, double[] outputs) {
        this.features = MatrixUtils.createRealVector(features);
        this.outputs = MatrixUtils.createRealVector(outputs);
    }

    public Example(double[] features, int label, int numLabels) {
        this.features = MatrixUtils.createRealVector(features);
        double arr[] = new double[numLabels];
        arr[label] = 1.0;
        this.outputs = MatrixUtils.createRealVector(arr);
    }

    public int getLabel() {
        double max = Double.NEGATIVE_INFINITY;
        int label = -1;
        double[] al = outputs.toArray();
        for (int i = 0; i < al.length; i++) {

            if (al[i] >= max) {
                max = al[i];
                label = i;
            }
        }
        return label;
    }

    public static int getLabel(RealVector outputs) {
        double max = Double.NEGATIVE_INFINITY;
        int label = -1;
        double[] al = outputs.toArray();
        for (int i = 0; i < al.length; i++) {

            if (al[i] >= max) {
                max = al[i];
                label = i;
            }
        }
        return label;
    }

    void print() {
        System.out.println(features + " mapped to " + outputs);
    }

    @Override
    public String toString() {
        String str = "";
        double[] al = features.toArray();
        for (double d : al) {
            str += d + ", ";
        }
        return "Example{" + "featureVector=" + str + ", class=" + getLabel(this.outputs) + '}' + "\n";
    }

}
