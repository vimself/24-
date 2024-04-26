from modeltrain import *
from DataUtils import *
from modeluse import *
import json

'''
在代码中的具体应用上通常
将 fit_transform 用于训练集的拟合和转换
将 transform 用于测试集或新数据的转换，以保证数据的一致性和正确的预处理操作
'''
with open(os.path.join("data", '1.txt'), 'r') as f:
    file_content = f.read()
    print(file_content)


