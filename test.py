from modeltrain import *
from DataUtils import *
from modeluse import *

'''
在代码中的具体应用上通常
将 fit_transform 用于训练集的拟合和转换
将 transform 用于测试集或新数据的转换，以保证数据的一致性和正确的预处理操作
'''

# getDataFromFile()

# print(getTokens("svision-online.de/mgfi/administrator/components/com_babackup/classes/fx29id1.txt"))
# model, vector=trainLR('data/data.csv')
model, vector = loadModel()
x = vector.transform(["askville.amazon.com/Lloyd-Klein-Jocelyn-Wildenstein-common/AnswerViewer.do?requestId=16406762"])
print(model.predict(x))