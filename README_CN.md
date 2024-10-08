# 基于WEKA的分类与特征选择
[English](README.md) | 中文

![](https://skillicons.dev/icons?i=java)

该库提供了三个经典数据集，使用**WEKA**学习和实践练习以决策树和神经网络为代表的机器学习算法，旨在将基础知识应用于数据挖掘、分类和特征提取等领域。

`script_weka`文件夹包含了基于Weka的四个基本算法代码（Java语言），可以作为学习Weka的基础和构建项目的基础，可根据实际需求对代码进行修改。

- 用于分类的决策树J48算法：[`J48`](https://github.com/Rc-W024/FeaExtrClassif-Weka/blob/main/script_weka/J48.java)

- 用于特征提取的神经网络算法：[`ANN`](https://github.com/Rc-W024/FeaExtrClassif-Weka/blob/main/script_weka/ANN.java)

- 用于聚类的K平均算法：[`K-Means`](https://github.com/Rc-W024/FeaExtrClassif-Weka/blob/main/script_weka/K-Mean.java)

- 用于回归分析的线性回归算法：[`Linear`](https://github.com/Rc-W024/FeaExtrClassif-Weka/blob/main/script_weka/Linear.java)

## WEKA
Weka是用于数据挖掘任务的机器学习算法的集合，通过Java开发，它包含了用于数据准备、分类、回归、聚类、关联规则挖掘和可视化的工具。Weka是根据**GNU通用公共许可证**发布的开源软件。此外，它还可用于构建机器学习、训练分类器和运行评估，且无需编写任何代码。

![image](https://user-images.githubusercontent.com/97808991/218695063-c454ba99-cdd6-4384-a744-a226ba848056.png)

更多信息，请访问：https://waikato.github.io/weka-site/index.html

## 决策树
### 示例：weather
所提供示例数据[`weather.arff`](https://github.com/Rc-W024/FeaExtrClassif-Weka/blob/main/weather.arff)的目标是预测我们今天是否可以打网球，其可用的训练数据包含了14天的气象数据。

![image](https://user-images.githubusercontent.com/97808991/218704353-6a665346-2e7d-473d-876d-2a019eb176f1.png)

文件共有14个具有5个属性的实例。选择每个属性会显示其一些特征：类型、可能值、最大值、最小值，以及显示该属性的样本分布和类分布的直方图。例如，图中显示属性“Outlook”具有三个可能值（*Sunny*、*Overcast*和*Rainy*），其分布为[5,4,5]。在属性*Outlook=sunny*的5个示例中，有3个为*no*类，2个为*yes*类。

加载数据后，我们可以学习模型，在本例中为决策树算法。我们可以选择如何评估模型：

- 使用训练集
- 提供的测试集
- 交叉验证
- 百分比分割

在本例中使用C4.5（J48）算法，随后选择*Use the training set*，点击*开始*按键。最后，我们可以使用*Visualize Tree*选项显示创建的树结构图。

![image](https://user-images.githubusercontent.com/97808991/218707658-aa22f5d6-3cc3-4d36-98f8-8f30fad7b016.png)

### 分类
**1. 药物选择**

在该例中（[`Drug1n.arff`](https://github.com/Rc-W024/FeaExtrClassif-Weka/blob/main/Drug1n.arff)），目标是根据不同的参数/变量预测要给予受过敏性鼻炎影响的患者药物。每个患者的临床记录中的变量如下：

- Age
- Sex
- Blood Pressure
- Cholesterol
- Na
- K

有5种备选药物，且药物适用于许多患者的数据已在多家医院收集。它旨在为新患者确定要使用的最佳药物。

加载`Drug1n.arff`中的数据，分析数据分布并使用J48创建决策树。可以使用与上述案例同样的方式对其分类精度进行评估。

![image](https://user-images.githubusercontent.com/97808991/218719736-4352e051-e87f-46d6-ae53-3b77f5b679f7.png)

**2. 地块分类**

在本例中（[`carac2008.arff`](https://github.com/Rc-W024/FeaExtrClassif-Weka/blob/main/carac2008.arff)），我们可以在包含1458个类别已知的地块的数据集上测试和评估决策树。通过Fetex提取44个特征来描述地块。

![image](https://user-images.githubusercontent.com/97808991/218713603-4e51e036-d4bf-45b9-b19b-afc611d0ee40.png)

在本例中，并不是所有的属性都用于分类（如`Id`），这表明存在对问题的解决不重要的属性。在如决策树之类的算法中，它们不会受到不重要属性的存在的严重影响，因为在学习过程中它们会选择属性。然而，其他分类器不执行属性选择，因此如果我们在学习之前执行属性过滤，我们可以显着提高它们的准确性，同时简化模型。

![image](https://user-images.githubusercontent.com/97808991/218721550-281ba30e-cb76-4f75-a8e5-cefd556fdccb.png)

移除`Id`后，使用J48算法得到的结果如下：

![image](https://user-images.githubusercontent.com/97808991/218722311-705f9454-1113-4faa-83dc-e0843b89e374.png)

## 人工神经网络
### weather
以上述[`weather.arff`](https://github.com/Rc-W024/FeaExtrClassif-Weka/blob/main/weather.arff)文件为例，加载文件中包含的数据，然后在“分类”选项卡中选择分类器。将选项GUI设定为开启后将打开一个界面，可在其中修改神经网络的体系结构。

![image](https://user-images.githubusercontent.com/97808991/218717996-e8ae9f07-871d-447e-868f-12a87e5ea35b.png)

![image](https://user-images.githubusercontent.com/97808991/218718289-484bbda1-cd26-463b-88d1-8fec4e7d7632.png)

### Drug1n
使用默认参数创建神经网络并用交叉验证对其进行评估。我们可以将结果与使用J48算法获得的结果进行比较。

![image](https://user-images.githubusercontent.com/97808991/218720471-45d35881-038f-4862-ac6b-ec98907af97f.png)

此外，我们还可以在图形界面中修改网络结构、添加新的神经元及其之间的连接。

### carac2008
加载文件中的数据并移除标识符对应的属性，使用默认参数创建神经网络。

![image](https://user-images.githubusercontent.com/97808991/218722539-abb6d92f-b2d9-4c99-8b28-a84fcca88a84.png)

## 参考信息
相关项目：基于机器学习算法的特征提取与分类面向地理空间情报数据库更新<br>(由瓦伦西亚理工大学*遥感与空间情报数据库更新*课目组织并管理)

开源数据集：https://waikato.github.io/weka-wiki/datasets/

开发者可使用Weka的Java API实现接口对接：[`Weka_API`](https://github.com/Rc-W024/Classification_Feature_Weka/blob/main/Weka_API.java)

WEKA深度学习：[WekaDeeplearning4j](https://github.com/Waikato/wekaDeeplearning4j/releases/tag/v1.7.2)
