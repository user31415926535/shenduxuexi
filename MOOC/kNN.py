import numpy as np
from math import sqrt
from collections import Counter
from sklearn import datasets

class KNN(object):

    iris=datasets.load_iris()
    X=iris.data
    y=iris.target
    def __init__(self,k):#初始化KNN分类器
        assert k>=1,"k必须大于等于1"
        self.k=k
        self._X_train=None#X_train前加_，将X_train私有化，用户不能随意操作
        self._y_train=None

    def fit(self,X_train,y_train):#根据X_train和y_train训练KNN数据集
        self._X_train=X_train
        self._y_train=y_train
        return self

    def predict(self,X_predict):#返回结果
        assert self._X_train is not None and self._y_train is not None,"X_train,y_train 不能是空"
        assert X_predict.shape[1]==self._X_train.shape[1],"传入参数特征个数必须与训练集相同"

        y_predict=[self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self,x):#给定单个预测数据，返回预测数据结果
        assert x.shape[0]==self._X_train.shape[1],"x的特征个数应该等于X_train里的特征个数"
        distances=[sqrt(np.sum((x_train-x)**2))for x_train in self._X_train]
        nearest=np.argsort(distances)
        topK_y=[self._y_train[i] for i in nearest[:self.k]]
        votes=Counter(topK_y)
        return votes.most_common(1)[0][0]
    def __repr__(self):
        return "KNN(k=%d)" % self.k
