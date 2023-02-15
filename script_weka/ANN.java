import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaNeuralNetworkFeatureExtractor {
  
  public static void main(String[] args) throws Exception {
    // Load the dataset
    DataSource source = new DataSource("iris.arff");
    Instances dataset = source.getDataSet();
    
    // Set the class index to the last attribute
    dataset.setClassIndex(dataset.numAttributes() - 1);
    
    // Configure and build the neural network
    MultilayerPerceptron mlp = new MultilayerPerceptron();
    mlp.setHiddenLayers("5");
    mlp.buildClassifier(dataset);
    
    // Output the feature values for each instance
    for (int i = 0; i < dataset.numInstances(); i++) {
      double[] features = mlp.distributionForInstance(dataset.instance(i));
      for (double feature : features) {
        System.out.print(feature + ",");
      }
      System.out.println();
    }
  }
}

