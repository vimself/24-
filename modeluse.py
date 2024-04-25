# -*- coding: utf-8 -*-
"""
    模型应用
"""
import pickle
from DataUtils import modelfile_path,vectorfile_path

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


