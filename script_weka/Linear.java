import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaLinearRegressionExample {
   public static void main(String[] args) throws Exception {
      DataSource source = new DataSource("data/housing.arff");
      Instances data = source.getDataSet();
      data.setClassIndex(data.numAttributes() - 1);
      
      LinearRegression model = new LinearRegression();
      model.buildClassifier(data);
      System.out.println(model);
   }
}

