import numpy as np
import matplotlib.pyplot as plt

plot_x=np.linspace(-1,6,141)

def dj(theta):
    return 2*(theta-2.5)
def j(theta):
    return (theta-2.5)**2-1
eta=0.1
epcilon=1e-8
theta=0.0
theta_history=[theta]
while True:
    gradient=dj(theta)
    last_theta=theta
    theta=theta-eta*gradient
    theta_history.append(theta)
    if(abs(j(theta)-j(last_theta))<epcilon):
        break

plt.plot(plot_x,j(plot_x))
plt.plot(np.array(theta_history),j(np.array(theta_history)),color='r',marker='+')
plt.show()


