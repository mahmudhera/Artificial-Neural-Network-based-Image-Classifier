package ann;

import java.io.FileInputStream;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.logging.Level;
import java.util.logging.Logger;
/*

 TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
 [offset] [type]          [value]          [description]
 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
 0004     32 bit integer  60000            number of items
 0008     unsigned byte   ??               label
 0009     unsigned byte   ??               label
 ........
 xxxx     unsigned byte   ??               label

 The labels values are 0 to 9.
 TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
 [offset] [type]          [value]          [description]
 0000     32 bit integer  0x00000803(2051) magic number
 0004     32 bit integer  60000            number of images
 0008     32 bit integer  28               number of rows
 0012     32 bit integer  28               number of columns
 0016     unsigned byte   ??               pixel
 0017     unsigned byte   ??               pixel
 ........
 xxxx     unsigned byte   ??               pixel

 Pixels are organized row-wise. 
 Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black). 
 */

public class MNISTLoader {

    public static final String TEST_LABEL = "t10k-labels.idx1-ubyte";
    public static final String TEST_IMAGE = "t10k-images.idx3-ubyte";
    
    public static final String TRAIN_LABEL = "train-labels.idx1-ubyte";
    public static final String TRAIN_IMAGE = "train-images.idx3-ubyte";
    public static final int NUM_CLASSES = 10;
    public static final int NUM_FEATURES = 784;

    private final String labelFile;
    private final String imageDataFile;

    public MNISTLoader(String labelFile, String imageDataFile) {
        this.labelFile = labelFile;
        this.imageDataFile = imageDataFile;
    }

    public ArrayList<Example> getExampleList() {
        ArrayList<Example> examples = new ArrayList<>();
        int[] labels = null;

        // reading label file
        try {
            FileInputStream labelFis = new FileInputStream(labelFile);
            byte[] bytes = new byte[4];

            //0000     32 bit integer  0x00000801(2049) magic number (MSB first)            
            labelFis.read(bytes, 0, 4);
            int magicNumber = (new BigInteger(bytes)).intValue();
            assert (magicNumber == 2049) : "Magic number must be 2049.";

            //0004     32 bit integer  60000            number of items
            labelFis.read(bytes, 0, 4);
            int numItems = (new BigInteger(bytes)).intValue();

            //0008, 0009, 0010.....     unsigned byte   =>    label
            labels = new int[numItems];
            for (int i = 0; i < numItems; i++) {
                labels[i] = labelFis.read();
                //System.out.println(labels[i]);
            }
            labelFis.close();
        } catch (Exception ex) {
            ex.printStackTrace();
            Logger.getLogger(MNISTLoader.class.getName()).log(Level.SEVERE, null, ex);
        }

        // reading image data file
        try {
            FileInputStream dataFis = new FileInputStream(imageDataFile);
            byte[] bytes = new byte[4];

            //0000     32 bit integer  0x00000801(2051) magic number (MSB first)            
            dataFis.read(bytes, 0, 4);
            int magicNumber = (new BigInteger(bytes)).intValue();
            //System.out.println(magicNumber);
            assert (magicNumber == 2051) : "Magic number must be 2051.";

            //0004     32 bit integer  60000            number of items
            dataFis.read(bytes, 0, 4);
            int numItems = (new BigInteger(bytes)).intValue();
            //assert numItems == 10000 or 60000;

            //0008     32 bit integer  28               number of rows
            dataFis.read(bytes, 0, 4);
            int numRows = (new BigInteger(bytes)).intValue();
            assert numRows == 28;

            //0012     32 bit integer  28               number of columns
            dataFis.read(bytes, 0, 4);
            int numCols = (new BigInteger(bytes)).intValue();
            assert numCols == 28;

            //0016, 0017, 0018, 0019.....     unsigned byte   =>    pixel
            int totalFeatures = numRows * numCols;
            double[] pixelValues;
            pixelValues = new double[totalFeatures];

            for (int item = 0; item < numItems; item++) {
                for (int j = 0; j < totalFeatures; j++) {
                    pixelValues[j] = (dataFis.read()) / 255.0;
                    // System.out.println(pixelValues[j]);
                }
                Example ex = new Example(pixelValues, getLabelArrayOneHot(labels[item], NUM_CLASSES));
                //ex.label = labels[item];
                examples.add(ex);
            }

            dataFis.close();
        } catch (Exception ex) {
            ex.printStackTrace();
            Logger.getLogger(MNISTLoader.class.getName()).log(Level.SEVERE, null, ex);
        }

        return examples;
    }

    public double[] getLabelArrayOneHot(int label, int numLabels) {
        double arr[] = new double[numLabels];
        arr[label] = 1.0;
        return arr;
    }

    public ArrayList<Example> getAllset(){
        ArrayList<Example> al = getExampleList();
        int s = 1000;
        ArrayList<Example> subset = new ArrayList<>();
        Collections.shuffle(al);
        for (int i = 0; i < s; i++) {
                subset.add(al.get(i));
        }
        return subset;
    }
    
    public ArrayList<Example> getCompleteSubset(int subestSize){
        ArrayList<Example> al = getExampleList();
        ArrayList<Example> subset = new ArrayList<>();
        HashMap<Integer, ArrayList<Example>> map = new HashMap<>();

        for(int i = 0; i < NUM_CLASSES; i++){
           // map.put(i, new ArrayList<>());
        }
        
        for (Example ex: al){
            map.get((Integer)(ex.getLabel())).add(ex);
        }
        
        int numExamplesPerClass = subestSize/10;
        for (Integer key: map.keySet()) {
            ArrayList<Example> keyExamples = map.get((Integer)key);
            for (int i = 0; i < numExamplesPerClass; i++) {
                subset.add(keyExamples.get(i));
            }
        }
        return subset;
    }
    
}

