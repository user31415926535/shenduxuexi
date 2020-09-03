import numpy as np

class simplelinearregression1(object):
    def __init__(self):#初始化模型
        self.a_=None
        self.b_=None

    def fit(self,x_train,y_train):
        #根据x_train，y_train训练模型
       
        assert len(x_train)==len(y_train),"训练集x_train必须和y_train长度一致"
        x_mean=np.mean(x_train)
        y_mean=np.mean(y_train)
        num=0.0
        d=0.0
        for x,y in zip(x_train,y_train):
            num+=(x-x_mean)*(y-y_mean)
            d+=(x-x_mean)**2

        self.a_=num/d
        self.b_=y_mean-self.a_*x_mean
        return self

    def predict(self,x_predict):
        #根据x_predict返回其结果
        assert x_predict.ndim==1,"维数必须为1"

        return [self._predict(x)for x in x_predict]


    def _predict(self,x_single):
        #根据待测数据x_single，返回其结果值
        return self.a_*x_single+self.b_


