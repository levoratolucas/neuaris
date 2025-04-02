import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


np.random.seed(1)
def plotmodel(w1,w2,b):
    x = np.linspace(-3,5,50) # gera um instevalio entre -2 e 4 com 50 valores
    y = (-w1*x - b)/w2


    plt.axvline(0,-1,1,color="green",linewidth=1)
    plt.axhline(0,-4,5,color="blue")
    plt.plot(x,y, color = "red")
    plt.grid(True)
plotmodel(w1=-3,w2=6,b=3)
x, y = make_classification(n_features=2, n_redundant=0,
                           n_informative=1, n_clusters_per_class=1)

plt.scatter(x[:, 0], x[:, 1], marker="o", c=y, edgecolors='k')

p=x[10]
plt.plot(p[0],p[1],marker='o' , color='r')
plt.show()
print(y[10])


print(x.shape, y.shape)
