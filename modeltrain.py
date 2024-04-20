# -*- coding: utf-8 -*-
"""
    模型训练

    在代码中的具体应用上通常
    将 fit_transform 用于训练集的拟合和转换
    将 transform 用于测试集或新数据的转换，以保证数据的一致性和正确的预处理操作

"""
import pandas as pd
import numpy as np
import random
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from DataUtils import getTokens,modelfile_path,vectorfile_path
from sklearn.metrics import classification_report

#从文件中获取数据集
def getDataFromFile(filename='data/data.csv'):
    input_url = filename
    data_csv = pd.read_csv(input_url, ',', error_bad_lines=False) #读取csv表格数据
    # print(data_csv)
    data_df = pd.DataFrame(data_csv) #将表格数据转换成DataFrame对象数据
    url_df = np.array(data_df) #将DF对象转换成NumPy数组
    # print(url_df)
    random.shuffle(url_df) #打乱数组顺序
    y = [d[1] for d in url_df] #取出每行第二个元素label
    inputurls = [d[0] for d in url_df] #取出每行第一个元素url
    return inputurls,y


#训练,通过逻辑回归模型训练
def trainLR(datapath):
    all_urls,y = getDataFromFile(datapath)
    url_vectorizer = TfidfVectorizer(tokenizer=getTokens)
    x = url_vectorizer.fit_transform(all_urls)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    l_regress = LogisticRegression()  # 实例化 Logistic regression
    l_regress.fit(x_train, y_train)   # 训练模型
    l_score = l_regress.score(x_test, y_test)
    print("score: {0:.2f} %".format(100 * l_score))
    print(classification_report(y_test,l_regress.predict(x_test)))
    return l_regress,url_vectorizer

#训练，通过SVM支持向量机模型训练
def trainSVM(datapath):
    all_urls, y = getDataFromFile(datapath)
    url_vectorizer = TfidfVectorizer(tokenizer=getTokens)
    x = url_vectorizer.fit_transform(all_urls)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    svmModel=svm.LinearSVC()
    svmModel.fit(x_train, y_train)
    svm_score=svmModel.score(x_test, y_test)
    print("score: {0:.2f} %".format(100 * svm_score))
    return svmModel,url_vectorizer

#保存模型及特征 序列化
def saveModel(model,vector):
    #保存模型
    file1 = modelfile_path
    with open(file1, 'wb') as f: #打开一个文件 file1 用于写入（'w'），并以二进制模式（'b'）打开
        pickle.dump(model, f) #使用 pickle 模块的 dump 函数将模型对象 model 序列化并写入文件对象 f 中
    f.close()
    #保存特征
    file2 = vectorfile_path
    with open(file2, 'wb') as f2:
        pickle.dump(vector, f2)
    f2.close()

if __name__ == '__main__':
    model,vector=trainLR('data/data.csv')
    #model, vector = trainSVM('data/data.csv')
    saveModel(model,vector)