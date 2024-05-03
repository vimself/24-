# 基于机器学习的恶意url识别系统

## 项目简介

本系统主要用于恶意URL识别，底层分别基于逻辑回归模型和SVM支持向量机实现的自动化恶意URL识别，20W条数据集源（good：100000，bad：110873）自于Alexa top-10w，PhishTank ，以及安全工具中已构造好的payload。全栈开发无数据库，前端vue2+bootstrap+echarts，后端Flask。

This system design is a graduation project for a student majoring in Network Engineering at Shenyang Ligong University. It is a cross design that combines machine learning and network security. The system version 3.0 is open for discussion and learning. Welcome to contact me proactively.

本项目的主要工作内容：

（1）分析恶意URL和正常URL的特征差异：识别系统将通过对大规模数据集进行分析，挖掘恶意URL和正常URL之间的特征差异

（2）建立基于机器学习的分类模型：将利用机器学习算法，构建恶意URL识别的分类模型，通过特征工程和模型优化技术，不断提升对恶意URL的检测精度和泛化能力，实现对恶意URL的快速准确识别

（3）设计与实现恶意URL识别系统：将设计并实现基于机器学习的恶意URL识别系统

## 项目运行环境

项目启动：运行backProcess.by文件，访问http://127.0.0.1:5000/

项目环境：python3，requirements.txt中有详细参数

```
$ pip3 install -r requirements.txt
```

## 项目运行效果

![image-20240429215408997](https://s2.loli.net/2024/04/29/1vWzlAQFnYbm6jO.png)

![image-20240429222243681](https://s2.loli.net/2024/04/29/I6nlgfx4zHpZwBY.png)

![image-20240429222318044](https://s2.loli.net/2024/04/29/VUZtP8nSpw69m35.png)

![image-20240429222335867](https://s2.loli.net/2024/04/29/6F3KgLcjJN87ktX.png)

![image-20240429222348703](https://s2.loli.net/2024/04/29/F8kjAtqfxQKwaem.png)

## 注：

我从一开始就想开源，因为我自己从0开发就很费事费力，所以不想让后人也重复做着没有很大价值的事情，并且为了提高投入产出比，希望机器学习研究者能将更多的精力放在底层算法模型的优化上，这也是我开源的最大原因。

本项目有许多提升空间，有好的idea，welcome to commit！