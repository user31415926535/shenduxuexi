import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from zijilianxi.lianxi1.train import train_list

if __name__=="__main__":
    iris=datasets.load_iris()
    x=iris.data
    y=iris.target
    x_train,y_train,x_test,y_test=train_list(x,y)


