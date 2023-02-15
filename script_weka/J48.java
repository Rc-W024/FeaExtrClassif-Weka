import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaJ48Example {
   public static void main(String[] args) throws Exception {
      DataSource source = new DataSource("data/iris.arff");
      Instances data = source.getDataSet();
      data.setClassIndex(data.numAttributes() - 1);
      
      J48 tree = new J48();
      tree.buildClassifier(data);
      System.out.println(tree);
   }
}

