# -*- coding: utf-8 -*-
"""
    模型训练

    在代码中的具体应用上通常
    将 fit_transform 用于训练集的拟合和转换
    将 transform 用于测试集或新数据的转换，以保证数据的一致性和正确的预处理操作

"""
import pandas as pd
import random
import pickle
from DataUtils import getTokens,modelfile_path,vectorfile_path
from sklearn import svm
from sklearn.linear_model import SGDClassifier
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report



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


def trainLR_outputData(datapath):
    all_urls, y = getDataFromFile(datapath)
    url_vectorizer = TfidfVectorizer(tokenizer=getTokens)
    x = url_vectorizer.fit_transform(all_urls)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    l_regress = LogisticRegression()  # 实例化 Logistic regression
    l_regress.fit(x_train, y_train)   # 训练模型
    l_score = l_regress.score(x_test, y_test)
    print("score: {0:.2f} %".format(100 * l_score))

    # 在模型上进行预测
    y_pred = l_regress.predict(x_test)

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    # 提取混淆矩阵中的相关值
    tn, fp, fn, tp = cm.ravel()

    # 计算指标
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_measure = 2 * (precision * recall) / (precision + recall)

    # 输出指标
    print("TPR (True Positive Rate): {0:.2f}".format(tpr))
    print("FPR (False Positive Rate): {0:.2f}".format(fpr))
    print("Precision: {0:.2f}".format(precision))
    print("Recall: {0:.2f}".format(recall))
    print("F-Measure: {0:.2f}".format(f_measure))

    return l_regress, url_vectorizer

#训练,通过逻辑回归模型训练,并画出loss曲线和acc曲线
def trainLR_DrawData(datapath):
    all_urls, y = getDataFromFile(datapath)
    url_vectorizer = TfidfVectorizer(tokenizer=getTokens)
    x = url_vectorizer.fit_transform(all_urls)

    # 将 'good' 和 'bad' 转换成二进制标签
    y = np.array([1 if yi == 'good' else 0 for yi in y])

    # 划分训练集和验证集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # 初始化并训练逻辑回归模型
    l_regress = LogisticRegression(solver='saga', max_iter=1, warm_start=True, tol=1e-5)

    # 记录损失和准确率的列表
    losses = []
    accuracies = []

    # 迭代次数，可以根据需要调整
    epochs = 50

    for epoch in range(epochs):
        l_regress.fit(x_train, y_train)

        # 预测概率
        probabilities = l_regress.predict_proba(x_train)

        # 计算log loss损失
        train_loss = log_loss(y_train, probabilities)
        losses.append(train_loss)

        # 计算准确率
        train_accuracy = l_regress.score(x_train, y_train)
        accuracies.append(train_accuracy)

        print(f'Epoch {epoch + 1}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}')

    # 测试模型并打印结果
    l_score = l_regress.score(x_test, y_test)
    print("Test score: {0:.2f} %".format(100 * l_score))
    print(classification_report(y_test, l_regress.predict(x_test)))

    # 绘制损失曲线
    plt.figure()
    plt.plot(np.arange(1, epochs + 1), losses, label='Train Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Log Loss')
    plt.legend()
    plt.savefig('data/logic_loss.png')

    # 绘制准确率曲线
    plt.figure()
    plt.plot(np.arange(1, epochs + 1), accuracies, label='Train Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('data/logic_accuracy.png')

    return l_regress, url_vectorizer



#训练，通过SVM支持向量机模型训练
def trainSVM(datapath):
    all_urls, y = getDataFromFile(datapath)
    url_vectorizer = TfidfVectorizer(tokenizer=getTokens)
    x = url_vectorizer.fit_transform(all_urls)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    svmModel = svm.LinearSVC()
    svmModel.fit(x_train, y_train)
    svm_score=svmModel.score(x_test, y_test)
    print("score: {0:.2f} %".format(100 * svm_score))
    return svmModel,url_vectorizer

def trainSVM_outputData(datapath):
    all_urls, y = getDataFromFile(datapath)
    url_vectorizer = TfidfVectorizer(tokenizer=getTokens)
    x = url_vectorizer.fit_transform(all_urls)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    svmModel = svm.LinearSVC()
    svmModel.fit(x_train, y_train)
    svm_score = svmModel.score(x_test, y_test)
    print("score: {0:.2f} %".format(100 * svm_score))

    # 在模型上进行预测
    y_pred = svmModel.predict(x_test)

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    # 提取混淆矩阵中的相关值
    tn, fp, fn, tp = cm.ravel()

    # 计算指标
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_measure = 2 * (precision * recall) / (precision + recall)

    # 输出指标
    print("TPR (True Positive Rate): {0:.2f}".format(tpr))
    print("FPR (False Positive Rate): {0:.2f}".format(fpr))
    print("Precision: {0:.2f}".format(precision))
    print("Recall: {0:.2f}".format(recall))
    print("F-Measure: {0:.2f}".format(f_measure))

    return svmModel, url_vectorizer

#训练，通过SVM支持向量机模型训练,并绘制loss曲线以及准确率曲线
def trainSVM_DrawData(datapath):
    all_urls, y = getDataFromFile(datapath)  # 假设这个函数返回所有URLs和对应标签
    url_vectorizer = TfidfVectorizer(tokenizer=getTokens)
    x = url_vectorizer.fit_transform(all_urls)
    for i in range(len(y)):
        if y[i] == "good":
            y[i] = 1
        if y[i] == "bad":
            y[i] = -1
    y = np.array(y).astype(int)  # 确保y是整数数组
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    sgdModel = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)

    # 记录loss和accuracy的列表
    losses = []
    accuracies = []

    # 迭代训练，更新loss和accuracy
    for epoch in range(1, 51):  # 这里只迭代50次，你可以根据需要调整迭代次数
        sgdModel.partial_fit(x_train, y_train, classes=np.unique(y))  # 更新模型
        # 使用决策函数计算margin (distance from the hyperplane) 来计算loss
        distances = sgdModel.decision_function(x_train)
        predicted = sgdModel.predict(x_train)
        # hinge loss 的简单实现
        hinge_loss = np.mean([max(0, 1 - yy * dist) for yy, dist in zip(y_train, distances)])
        losses.append(hinge_loss)
        # 计算accuracy
        accuracy = accuracy_score(y_train, predicted)
        accuracies.append(accuracy)
        print(f'Epoch {epoch}, Loss: {hinge_loss}, Accuracy: {accuracy}')

    svm_score = accuracy_score(y_test, sgdModel.predict(x_test))
    print("Test score: {0:.2f} %".format(100 * svm_score))

    # 绘制loss曲线
    plt.figure()
    plt.plot(range(1, 51), losses, label='Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Hinge Loss')
    plt.legend()
    plt.savefig('data/svm_loss.png')  # 保存loss曲线图像

    # 绘制accuracy曲线
    plt.figure()
    plt.plot(range(1, 51), accuracies, label='Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('data/svm_accuracy.png')  # 保存accuracy曲线图像

    return sgdModel,url_vectorizer

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
    # model,vector=trainLR('data/data.csv')
    # model, vector = trainLR_outputData('data/data.csv')
    # model,vector=trainLR_DrawData('data/data.csv')

    # model, vector = trainSVM('data/data.csv')
    model, vector = trainSVM_outputData('data/data.csv')
    # model, vector = trainSVM_DrawData('data/data.csv')
    saveModel(model,vector)