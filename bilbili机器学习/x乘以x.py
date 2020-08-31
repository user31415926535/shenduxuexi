import numpy as np
import matplotlib.pyplot as plt 
import math
if __name__=="__main__":
    x=np.arange(0,1000,0.1)
    y=[a**a for a in x]
    plt.plot(x,y,linewidth=2,color="#007500",label="y=x^x")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()



