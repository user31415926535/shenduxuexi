import math
import numpy as np
import matplotlib.pyplot as plt
from MOOC.xianxinghuigui.simplelinearregression import simplelinearregression1
x=np.array([1.,2.,3.,4.,5.])
y=np.array([1.,3.,2.,3.,5.])

x_predict=6
reg1=simplelinearregression1()
reg1.fit(x,y)
ss=reg1.predict(np.array([x_predict]))
print(ss)






