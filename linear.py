# equação da reta 
# ax + by + c

import numpy as np
import matplotlib.pyplot as plt


a = -1
b = 4
c = 0.4

# a*x + b*y + c
# y = (-a*x - c)/b
def plotline(a,b,c):
    x = np.linspace(-2,4,50) # gera um instevalio entre -2 e 4 com 50 valores
    y = (-a*x - c)/b


    plt.axvline(0,-1,1,color="green",linewidth=1)
    plt.axhline(0,-2,4,color="blue")
    plt.plot(x,y, color = "red")
    plt.grid(True)


p1 = (2 , 0.4) # solucionanod eqaução da reta no ponto x ,y designado
p2 = (1 , 0.6) # solucionanod eqaução da reta no ponto x ,y designado
# a*x + b*y + c
# a*p1[0] + b*p1[1] + c

ret = a*p1[0] + b*p1[1] + c
ret2 = a*p2[0] + b*p2[1] + c
print("%.2f" % ret , "%.2f" % ret2)

# 0.00 1.80

plotline(a,b,c)
plt.plot(p1[0],p1[1],marker = "o", color = "blue")
plt.plot(p2[0],p2[1],marker = "o", color = "red")

plt.show()

