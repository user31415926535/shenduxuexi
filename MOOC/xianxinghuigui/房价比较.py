import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import sklearn
from MOOC.xianxinghuigui.model_selection import train_list
from MOOC.xianxinghuigui.simplelinearregression import simplelinearregression1
boston=datasets.load_boston()
x=boston.data[:,5]
y=boston.target
x=x[y<50]
y=y[y<50]
x_train,x_test,y_train,y_test=train_list(x,y,seed=666)

reg=simplelinearregression1()
reg.fit(x_train,y_train)


plt.scatter(x_train,y_train)
plt.plot(x_train,reg.predict(x_train),color='red')
plt.show()

