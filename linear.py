# equação da reta 
# ax + by + c
# a e b são pesos assim sendo e o c é o vies
# y =  w1* x1 + w2*x2 + b ############ duas dimensões
# y =  w1* x1 + w2*x2 + w3 * x3 + b ### três dimensões

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
p3 = (3 , -0.2) # solucionanod eqaução da reta no ponto x ,y designado
# a*x + b*y + c
# a*p1[0] + b*p1[1] + c

ret = a*p1[0] + b*p1[1] + c
ret2 = a*p2[0] + b*p2[1] + c
ret3 = a*p3[0] + b*p3[1] + c
print("%.2f\n" % ret , "%.2f\n" % ret2,"%.2f" % ret3)

# 0.00
# 1.80
# -3.40

plotline(a,b,c)
plt.plot(p1[0],p1[1],marker = "o", color = "blue")
plt.plot(p2[0],p2[1],marker = "o", color = "red")
plt.plot(p3[0],p3[1],marker = "o", color = "g")

plt.show()

