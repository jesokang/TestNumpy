from numpy import *
import operator

#数据集建立。group为数据，labels为标签
def createDataSet():
    group= array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group, labels

#将文档数据转化为矩阵数据
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readline()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classlabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classlabelVector.append(int(listFromLine[-1]))
        index +=1
    return returnMat,classlabelVector

#KNN算法
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat =diffMat **2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistInicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistInicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)#利用sorted函数来对计算出来的数据进行排序
    return sortedClassCount[0][0]


# help(sorted)
