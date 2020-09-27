import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score
import numpy as np
import math 
from MOOC.kjinlin.kNN import KNN
from MOOC.kjinlin.model_selection import train_list
#from MOOC.matris import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
if __name__=="__main__":
    digits=datasets.load_digits()
    print(digits.keys())
    X=digits.data
    y=digits.target#从sklearn调用数据并把数据和结果分别传给X和y
    d_test=train_list(X,y)

    #KNNclf=KNN(k=6)
    #KNNclf.fit(X_train,y_train)
    #y_predict=KNNclf.predict(X_test)
    #result=accuracy_score(y_test,y_predict)
    #print(result)
    #some_digits=X[666]
    #some_digits_image=some_digits.reshape(8,8 )
    #plt.imshow(some_digits_image,cmap=matplotlib.cm.binary)
    #plt.show()

    best_k=-1
    best_p=-1
    best_score=0.0
    for k in range(1,11):
        for p in range (1,6):
            KNNclf=KNeighborsClassifier(n_neighbors=k,weights="distance")
            KNNclf.fit(X_train,y_train)
            score=KNNclf.score(X_test,y_test)
            if score>best_score:
                best_k=k
                best_p=p
                best_score=score
    print("best_k=",best_k)
    print("best_p=",best_p)
    print("best_core=",best_score)