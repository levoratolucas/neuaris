import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


np.random.seed(30)


def plotmodel(w1, w2, b , X,Y):
    # gera um instevalio entre -2 e 4 com 50 valores
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=Y, edgecolors='k')
    
    xmin, xmax = plt.gca().get_xlim()
    ymin, ymax = plt.gca().get_ylim()
    
    x = np.linspace(-3, 5, 50)
    y = (-w1*x - b)/w2

    plt.axvline(0, -1, 1, color="green", linewidth=1)
    plt.axhline(0, -2, 4, color="blue")
    plt.plot(x, y, color="red")
    plt.grid(True)
    
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)


# pesos e viéses pra alterar a reta
w1 = 0.03
w2 = 1.32
b = -0.2
############################

X, Y = make_classification(n_features=2, n_redundant=0,
                           n_informative=1, n_clusters_per_class=1)

p = (0,2)

def classify(ponto , w1 , w2 , b):
    ret = w1 * ponto[0] + w2 * ponto[1] + b
    
    if ret >= 0: 
        return 1, "yellow"
    else:
        return 0,"blue"
def classify(ponto , w1 , w2 , b):
    ret = w1 * ponto[0] + w2 * ponto[1] + b
    
    if ret >= 0: 
        return 1, "yellow"
    else:
        return 0,"blue"



p = (2,-2)
classe , color = classify(p,w1,w2,b)

print(classe,color)
plotmodel(w1, w2, b, X, Y)

plt.plot(p[0],p[1],marker = 'o', color = 'r')



plt.show()

acertos = 0
for k in range(len(X)):
    categ, _ = classify(X[k], w1,w2,b)
    if categ == Y[k]:
        acertos+=1
        
print("acuracy: {0} %".format((acertos/len(X))*100))