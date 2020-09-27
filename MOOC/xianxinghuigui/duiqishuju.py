import numpy as np

def duiqishuju(x_train,y_train):
    y_train.reshape(-1,1)
    lenx_train=len(x_train)
    leny_train=len(y_train)
    thelist=[(x_train),(y_train)]
    themartix=[[x_train,y_train]]
    max_index=np.argmax(themartix,1).squeeze(0)
    min_index=np.argmin(themartix,1).squeeze(0)
    pad_len = np.max([lenx_train, leny_train]) - np.min([lenx_train, leny_train])
    arr_min = thelist[min_index]
    pad_arr = np.pad(arr_min, (0, pad_len), 'constant', constant_values=0)
    return pad_arr, thelist[max_index]


