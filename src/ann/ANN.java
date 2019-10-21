package ann;

import java.util.ArrayList;
import java.util.Collections;

/**
 *
 * @author Hera
 */
public class ANN {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        
        ArrayList<Example> entireDataset = MNISTDatasetTrain();
        
        Collections.shuffle(entireDataset);
        ArrayList<Example> trainingSet = new ArrayList<>();
        ArrayList<Example> validationSet;
        ArrayList<Example> testSet;
        
        int n = entireDataset.size();
        for (int i = 0; i < n * 0.85; i++) {
            trainingSet.add(entireDataset.remove(0));
        }
        validationSet = new ArrayList(entireDataset);
        testSet = MNISTDatasetTest();
        
        //int[] layers = {AbaloneLoader.NUM_FEATURES, 8, AbaloneLoader.NUM_CLASSES};
        int [] layers = {MNISTLoader.NUM_FEATURES, 26, MNISTLoader.NUM_CLASSES};
        int minibatchsize = 1;

        Network net = constructNetwork(trainingSet, validationSet, layers, minibatchsize, 100, 1, 500, 1.3);
        double accuracy = net.check(testSet);
        
        System.out.println(accuracy);

    }
    
    public static Network constructNetwork(ArrayList<Example> trainingSet, ArrayList<Example> validationSet, 
                            int [] layerSizes, int minibatchsize,
                            double targetAccuracy, int MAX_STEPS, int numEpochsInEachStep, double learningRate) {
       
        double bestAccuracy = -1;
        Network bestNetwork = null;
        int iterationCount = 0;
       
        while (bestAccuracy < targetAccuracy && iterationCount < MAX_STEPS) {
            iterationCount++;
            System.out.println("Iteration count: " + iterationCount);
            Network net = new Network (trainingSet, layerSizes, minibatchsize);
            net.train(numEpochsInEachStep, learningRate);
            double accuracy = net.check(validationSet);
            System.out.println("Accuracy now on val: "+ accuracy);
            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                bestNetwork = net;
            }
            for (int i = 1; i < layerSizes.length-1; i++) {
                layerSizes[i]++;
                layerSizes[i]++;
            }
        }
        
        if (bestNetwork == null)
            return null;
       
        return bestNetwork;
       
    }
    
    static ArrayList<Example> MNISTDatasetTest() {
        MNISTLoader loader = new MNISTLoader(MNISTLoader.TEST_LABEL, MNISTLoader.TEST_IMAGE);
        ArrayList<Example> al = loader.getExampleList();
        return al;
    }
    
    static ArrayList<Example> MNISTDatasetTrain() {
        MNISTLoader loader = new MNISTLoader(MNISTLoader.TRAIN_LABEL, MNISTLoader.TRAIN_IMAGE);
         ArrayList<Example> al = loader.getAllset();
        return al;
    }

}
