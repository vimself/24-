# -*- coding: utf-8 -*-
"""
    模型应用
"""
from flask import Flask
import pickle
from DataUtils import modelfile_path,vectorfile_path

app = Flask(__name__)

#载入已经训练好的模型 反序列化
def loadModel():
    file1 = modelfile_path
    with open(file1, 'rb') as f1: # 打开一个文件 file1 用于读取（'r'），并以二进制模式（'b'）打开
        model = pickle.load(f1) # 使用 pickle 模块的 load 函数从文件对象 f1 中反序列化并加载模型对象
    f1.close()

    file2 = vectorfile_path
    with open(file2, 'rb') as f2:
        vector = pickle.load(f2)
    f2.close()
    return model,vector

#通过接口进行调用
'''
@app.route('/<path:path>') 是一个装饰器
它告诉 Flask 在处理指定路径的 HTTP 请求时应该调用下面的函数
在这个例子中，路径是 <path:path>，意味着它可以匹配任何路径，并将路径作为参数传递给下面的函数
'''
@app.route('/<path:path>')
def show_predict(path):
    X_predict = []
    X_predict.append(path)
    model, vector = loadModel()
    x = vector.transform(X_predict)
    print(x)
    y_predict = model.predict(x)
    print(y_predict)
    print("-------------")
    return "url predict: "+str(y_predict[0])


'''
这是 Python 的惯用法，用来检查当前模块是否是被直接执行的
如果是，则执行 app.run()，启动 Flask 应用的服务器
'''
if __name__ == "__main__":
	 app.run()

