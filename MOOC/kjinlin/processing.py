import numpy as np
class standerscaler:
    def __init__(self):
        np.mean_=None
        np.scale_=None #scale_相当于std_
    def fit(self,X):
        #根据传入数据求出均值与方差
        assert X.ndim==2,"传入数据必须是二维的"
        self.mean_=np.array(np.mean[:,i] for i in range(X.shape[1]))
        self.scale_=np.array(np.std[:,i]for i in range(X.shape[1]))
        return self
    def transform(self,X):
        assert self.mean_ is not None and self.scale_ is not None ,"均值和方差不能为空"
        assert X.shape[1]==len(self.mean_),"数组个数必须一致"
        assert X.ndim==2,"数据必须是二维的"
        resX=np.empty(X,dtype=float)
        for col in range(X.shape[1]):
            resX[:,col]=(X[:,col]-np.mean[:,col])/np.std[:,col]
        return resX


