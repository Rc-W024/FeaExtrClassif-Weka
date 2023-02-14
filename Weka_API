DataSource source = new DataSource("/some/where/data.arff");
Instances data = source.getDataSet();

J48 tree = new J48();
Evaluation eval = new Evaluation(data);
eval.crossValidateModel(tree, data, 10, new Random(1));

