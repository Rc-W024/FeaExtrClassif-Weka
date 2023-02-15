import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaKMeansExample {
   public static void main(String[] args) throws Exception {
      DataSource source = new DataSource("data/iris.arff");
      Instances data = source.getDataSet();
      data.setClassIndex(data.numAttributes() - 1);
      
      SimpleKMeans model = new SimpleKMeans();
      model.setNumClusters(3);
      model.buildClusterer(data);
      System.out.println(model);
   }
}

