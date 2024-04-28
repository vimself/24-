# -*- coding: utf-8 -*-
"""
    数据处理
    洗词,分词
"""
import os

basedir=os.path.abspath(os.path.dirname(__file__)) #获取该脚本所在目录的绝对路径
model_path=os.path.join(basedir,'model') #构建一个model子目录的路径
modelfile_path=os.path.join(model_path,'model.pkl') #在model子目录下创建model.pkl文件
vectorfile_path=os.path.join(model_path,'vector.pkl') #在model子目录下创建vector.pkl文件


'''
    / 分割
'''
def slash_split(text):
    index = 0
    while True:
        str = text[index]
        for i in range(len(str)):
            if str[i] == '/':
                if i-1 == -1:
                    text.append(str[i+1:])
                    text.remove(str)
                    break
                if i+1 == len(str) and str[i] == '/' and str[i-1] != '/':
                    text.append(str[:i])
                    text.remove(str)
                    break
                if str[i-1] == '/' or str[i+1] == '/':
                    continue
                if i-2 >= 0 and str[i-1] == '.' and str[i-2] == '.':
                    continue
                if str[i+1] == '&':
                    continue
                if str[i+1] == '>':
                    continue
                if str[i+1] == '?' and i+2 < len(str):
                    text.append(str[:i])
                    text.append(str[i+2:])
                    text.remove(str)
                    break
                text.append(str[:i])
                text.append(str[i+1:])
                text.remove(str)
                break
        if index + 1 < len(text):
            index += 1
        else:
            break
    return text

'''
    ? 分割
'''
def question_split(text):
    index = 0
    while True:
        str = text[index]
        for i in range(len(str)):
            if str[i] == '?':
                if i - 1 == -1:
                    text.append(str[i + 1:])
                    text.remove(str)
                    break
                if i + 1 == len(str) and str[i] == '?' and str[i - 1] != '?':
                    text.append(str[:i])
                    text.remove(str)
                    break
                if str[i - 1] == '?' or str[i + 1] == '?':
                    continue
                text.append(str[:i])
                text.append(str[i + 1:])
                text.remove(str)
                break
        if index + 1 < len(text):
            index += 1
        else:
            break
    return text

'''
    & 分割
'''
def address_split(text):
    index = 0
    while True:
        str = text[index]
        for i in range(len(str)):
            if str[i] == '&':
                if i - 1 == -1:
                    text.append(str[i + 1:])
                    text.remove(str)
                    break
                if i + 1 == len(str) and str[i] == '&' and str[i - 1] != '&':
                    text.append(str[:i])
                    text.remove(str)
                    break
                if str[i - 1] == '&' or str[i + 1] == '&':
                    continue
                if str[i-1] == '/':
                    text.append(str[:i-1])
                    text.append(str[i+1:])
                    text.remove(str)
                    break
                text.append(str[:i])
                text.append(str[i + 1:])
                text.remove(str)
                break
        if index + 1 < len(text):
            index += 1
        else:
            break
    return text

'''
    <br/> 分割
'''
def br_split(text):
    res = []
    for i in range(len(text)):
        arr = text[i].split("<br/>")
        res = res + arr
    return res

def amp_split(text):
    res = []
    for i in range(len(text)):
        arr = text[i].split("&amp;")
        res = res + arr
    return res


def separation(arr):
    result1 = br_split(arr)
    result2 = amp_split(result1)
    result3 = slash_split(result2)
    result4 = question_split(result3)
    result5 = address_split(result4)
    return result5

# 分词
def getTokens(input):
    web_url = input.lower() #字符串转小写
    res = separation([web_url])
    urltoken = list(set(res))
    #分割字符串，将'/'和'-'分割开，'.'包含在字符串内，然后进行各种排列组合，得出多种分词
    if 'com' in urltoken:
        urltoken.remove('com')
    if 'cn' in urltoken:
        urltoken.remove('cn')
    if 'example' in urltoken:
        urltoken.remove('example')
    if 'www' in urltoken:
        urltoken.remove('www')
    #去除com cn 等无影响字符串
    return urltoken

def data_count(input):
    web_url = input.lower()
    dict = [{"@": 0, "eval": 0, "exec": 0, "<": 0, ">": 0, "system": 0, "echo": 0, "script": 0, "javascript": 0, "by": 0,
            "order": 0, "or": 0, "and": 0, "select": 0, "perl": 0, "php": 0, "net": 0, "nc": 0, "on": 0, "sleep": 0,
            "svg": 0, "alert": 0, "union": 0, "group": 0},{"/":0,"?":0,"&":0,"<br/>":0,"&amp;":0,".":0,"-":0,"_":0},0]
    for i in range(2):
        for key in dict[i]:
            dict[i][key] = web_url.count(key)

    dict[2] = len(web_url)
    return dict

# print(data_count("1476510729.xiazaidown.com/cx/160624/6/hplaserjet1020@151_974<br/>.exe"))

