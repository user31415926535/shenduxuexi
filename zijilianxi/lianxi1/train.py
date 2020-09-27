import numpy as np
def train_list(x,y,test_rato=0.2,seed=None):
    if seed:
        np.random.seed(seed)
    suijidaluan=np.random.permutation(len(x))
    the_size=int(len(x)*test_rato)
    ceshishuju=suijidaluan[:the_size]
    xunlianshuju=suijidaluan[the_size:]
    x_train=x[xunlianshuju]
    y_train=y[xunlianshuju]

    x_test=x[ceshishuju]
    y_test=y[ceshishuju]

    return x_train,y_train,x_test,y_test
