# -*- coding: utf-8 -*-
"""
    数据处理
"""
import os
basedir=os.path.abspath(os.path.dirname(__file__)) #获取该脚本所在目录的绝对路径
model_path=os.path.join(basedir,'model') #构建一个model子目录的路径
modelfile_path=os.path.join(model_path,'model.pkl') #在model子目录下创建model.pkl文件
vectorfile_path=os.path.join(model_path,'vector.pkl') #在model子目录下创建vector.pkl文件
# 分词
def getTokens(input):
    web_url = input.lower() #字符串转小写
    urltoken = []
    dot_slash = []
    slash = str(web_url).split('/')
    for i in slash:
        r1 = str(i).split('-')
        token_slash = []
        for j in range(0, len(r1)):
            r2 = str(r1[j]).split('.')
            token_slash = token_slash + r2
        dot_slash = dot_slash + r1 + token_slash
    urltoken = list(set(dot_slash))
    #分割字符串，将'/'和'-'分割开，'.'包含在字符串内，然后进行各种排列组合，得出多种分词
    if 'com' in urltoken:
        urltoken.remove('com')
    if 'cn' in urltoken:
        urltoken.remove('cn')
    #去除com cn 等无影响字符串
    return urltoken
