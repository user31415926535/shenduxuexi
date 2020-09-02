import numpy as np
import matplotlib.pyplot as plt
#均值最值归一化
x=np.random.randint(0,100,(50,10))
x=np.array(x,dtype=float)
for i in range(0,10):

    x[:,i]=(x[:,i]-np.min(x[:,i]))/(np.max(x[:,i])-np.min(x[:,i]))
    print("x[50,",i,"]=")
    print(x[:,i])
plt.scatter(x[:,0],x[:,1],color='red')
plt.show()    


#均值方差归一化
x2=np.random.randint(0,100,(50,10))
x2=np.array(x,dtype=float)
for t in range(0,10):
    x2[:,t]=(x[:,t]-np.mean(x2[:,t]))/np.std(x[:,t])
    print("x2[50,",t,"]=")
    print(x[:,i])
plt.scatter(x2[:,0],x2[:,1],color="blue")
plt.show()

