from modeltrain import *
from DataUtils import *
from modeluse import *
import json

'''
在代码中的具体应用上通常
将 fit_transform 用于训练集的拟合和转换
将 transform 用于测试集或新数据的转换，以保证数据的一致性和正确的预处理操作
'''
import csv
import io

# 打开 CSV 文件
with open('data/data.csv', mode='r',encoding='latin-1') as file:
    reader = csv.reader(file)
    rows = list(reader)
    # print(rows)

goodnum = 0
badnum = 0
# 将每一行的第一个值添加 "good"
for row in rows:
    if row[0].startswith("http://"):
        row[0] = row[0].replace("http://","",1)
    if row[0].startswith("https://"):
        row[0] = row[0].replace("https://","",1)

print(goodnum)
print(badnum)

# 保存修改后的数据到新的 CSV 文件
with open('data/new.csv', mode='w', newline='',encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(rows)

