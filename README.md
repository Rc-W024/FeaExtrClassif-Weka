# Classification and Feature Selection with WEKA
English | [中文](README_CN.md)

Three classic datasets are provided for using **WEKA** to study and practice machine learning (ML) algorithms such as Decision Tree and Artificial Neural Network (ANN), aiming to apply basic knowledge to areas such as data mining and classification, and feature extraction.

The `script_weka` folder contains four basic algorithm codes (*Java*) for Weka, which can be used as the basis for learning Weka or as the basis for projects.

- Decision Tree (J48) for classification [`J48`](https://github.com/Rc-W024/FeaExtrClassif-Weka/blob/main/script_weka/J48.java)

- ANN for feature extraction [`ANN`](https://github.com/Rc-W024/FeaExtrClassif-Weka/blob/main/script_weka/ANN.java)

- K-means for clustering [`K-Means`](https://github.com/Rc-W024/FeaExtrClassif-Weka/blob/main/script_weka/K-Mean.java)

- Linear regression for regression analysis [`Linear`](https://github.com/Rc-W024/FeaExtrClassif-Weka/blob/main/script_weka/Linear.java)

## WEKA
Weka is a collection of ML algorithms for data mining tasks, which is developed in **Java**. It contains tools for data preparation, classification, regression, clustering, association rules mining, and visualization. It is open source software issued under the [GNU General Public License](https://www.gnu.org/licenses/gpl.html).

In addition, Weka can be used to build ML pipelines, train classifiers, and run evaluations without having to write a single line of code.

![image](https://user-images.githubusercontent.com/97808991/218695063-c454ba99-cdd6-4384-a744-a226ba848056.png)

For more information, please access: https://waikato.github.io/weka-site/index.html

## Decision Tree
### Example of weather
The objective of [`weather.arff`](https://github.com/Rc-W024/FeaExtrClassif-Weka/blob/main/weather.arff) is to predict if today we can play tennis. The available training data contain meteorological data of 14 days.

![image](https://user-images.githubusercontent.com/97808991/218704353-6a665346-2e7d-473d-876d-2a019eb176f1.png)

There are 14 instances with 5 attributes. Selecting each attribute, some characteristics of the attribute are displayed: type (nominal/numeric), possible values, maximum and minimum values and a histogram showing the distribution of the samples for this attribute and the classes distribution. For example, the figure shows that the attribute *Outlook* has three possible values (Sunny, Overcast and Rainy) having a distribution of [5,4,5]. In the 5 samples where the attribute *Outlook=sunny*, there are 3 with the class *no* and 2 with class *yes*.

Once the data are loaded, we can learn a model, a decision tree in this case. We can choose how to evaluate the model:

- Use training set
- Supplied test set
- Cross-validation
- Percentage split

In this case we are going to use the algorithm C4.5 (J48 is its name in Weka). Then, select *Use the training set* and press the button *Start*. We can display the created tree using the option *Visualize Tree*.

![image](https://user-images.githubusercontent.com/97808991/218707658-aa22f5d6-3cc3-4d36-98f8-8f30fad7b016.png)

### Classfication
**1. Drug selection**

In this case ([`Drug1n.arff`](https://github.com/Rc-W024/FeaExtrClassif-Weka/blob/main/Drug1n.arff)), the objective is to predict the drug to be given to a patient affected by allergic rhinitis according to different parameters/variables. The variables in the clinical records of each patient are:

- Age
- Sex
- Blood Pressure
- Cholesterol
- Na
- K

There are 5 alternative drugs: DrugA, DrugB, DrugC, DrugX, DrugY. The data of the drug suitable for many patients have been collected in several hospitals. It is intended, for new patients, to determine the best drug to be used.

We can load the data in `Drug1n.arff`, analyze the data distribution and create a decision tree with J48. The classification accuracy can be evaluated in the same way as the above case.

![image](https://user-images.githubusercontent.com/97808991/218719736-4352e051-e87f-46d6-ae53-3b77f5b679f7.png)

**2. Parcel Classification**

In this case ([`carac2008.arff`](https://github.com/Rc-W024/FeaExtrClassif-Weka/blob/main/carac2008.arff)), we can test the decision trees over a dataset with 1458 parcels whose classes are known. The parcels are described by 44 features extracted with the software [Fetex](https://cgat.webs.upv.es/BigFiles/fetex2web/demofetex/SoftwareFETEX.pdf).

![image](https://user-images.githubusercontent.com/97808991/218713603-4e51e036-d4bf-45b9-b19b-afc611d0ee40.png)

In this case, not all the attributes are used to make a classification such as `Id`, this indicates that there are attributes that are not significant for the resolution of the problem. There are methods such as decision trees, which are not seriously affected by the presence of non-significant attributes, since in the learning process they make a selection of attributes. However, other classifier methods do not perform the attribute selection, so if we perform a filtering of attributes prior to learning we can significantly improve their accuracy, and at the same time simplify the models.

![image](https://user-images.githubusercontent.com/97808991/218721550-281ba30e-cb76-4f75-a8e5-cefd556fdccb.png)

After deleting the identifier (`Id`), the result obtained by using the J48 algorithm is as follows:

![image](https://user-images.githubusercontent.com/97808991/218722311-705f9454-1113-4faa-83dc-e0843b89e374.png)

## Artificial Neural Network
### weather
Load the data contained in the file [`weather.arff`](https://github.com/Rc-W024/FeaExtrClassif-Weka/blob/main/weather.arff), and select the classifier in the tab “Classify”. In the classifier options, select `GUI=True`. This option opens an interface where the architecture of the neural network can be modified.

![image](https://user-images.githubusercontent.com/97808991/218717996-e8ae9f07-871d-447e-868f-12a87e5ea35b.png)

![image](https://user-images.githubusercontent.com/97808991/218718289-484bbda1-cd26-463b-88d1-8fec4e7d7632.png)

### Drug1n
Create a neural network with the parameters by default and evaluate it with cross validation. We can compare the results with the obtained with the J48 algorithm.

![image](https://user-images.githubusercontent.com/97808991/218720471-45d35881-038f-4862-ac6b-ec98907af97f.png)

In addition, we can also modify the arquitecture of the network in the graphical interface to add new neurons and connections between them.

### carac2008
Load the data in the file and remove the attribute corresponding to the identifier, and create a neural network with the default parameters.

![image](https://user-images.githubusercontent.com/97808991/218722539-abb6d92f-b2d9-4c99-8b28-a84fcca88a84.png)

## FYI
Related project: Feature Extraction and Classification based on ML Algorithms for Land Use Database Updating<br />(Developed by the *Remote Sensing and Geodatabase Updating* subject of UPV)

OA Dataset: https://waikato.github.io/weka-wiki/datasets/

Programmers can easily implement the pipeline using Weka's Java API: [`Weka_API`](https://github.com/Rc-W024/Classification_Feature_Weka/blob/main/Weka_API.java)

Deep Learning with WEKA: [WekaDeeplearning4j](https://github.com/Waikato/wekaDeeplearning4j/releases/tag/v1.7.2)
