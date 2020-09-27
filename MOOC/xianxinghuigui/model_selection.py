import numpy as np
from sklearn import datasets
def train_list(X,y,test_rato=0.2,seed=None):#将X，y分割成X_train，y_train，X_test，y_test
    assert X.shape[0]==y.shape[0],"the size of X must be equal to y"
    assert 0.0<=test_rato<=1.0,"test_rato must be availd "

    if seed:
        np.random.seed(seed)

    shuffled_indexes=np.random.permutation(len(X))
    test_size=int(len(X)*test_rato)
    test_indexes=shuffled_indexes[:test_size]
    train_indexes=shuffled_indexes[test_size:]

    X_train=X[train_indexes]
    y_train=y[train_indexes]

    X_test=X[test_indexes]
    y_test=y[test_indexes]
    return X_train,X_test,y_train,y_test
