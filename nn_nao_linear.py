import torch
from torch import nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.datasets import make_classification


np.random.seed(30)
# pesos e viéses pra alterar a reta
w1 = 0.03
w2 = 1.32
b = -0.2
X, Y = make_classification(n_features=2, n_redundant=0,
                           n_informative=1, n_clusters_per_class=1)


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



perceptron = nn.Linear(2,1)
# activ = nn.Sigmoid()
activ = nn.Tanh()
# activ = nn.ReLU()
perceptron.weight = nn.Parameter(torch.Tensor([[w1,w2]]))
perceptron.bias = nn.Parameter(torch.Tensor([b]))

colors = ['r','b','y','g']
markers = [ '^','<','>','v']
plt.figure()

for k, idx in enumerate([17,21,43,66]):
    x = torch.Tensor(X[idx])
    
    ret = perceptron(x)
    act = activ(ret)
    
    act_linear = 0 if ret.data < 0 else 1
    
    label = "ret: {:5.2f} ".format(ret.data.numpy()[0])+"limiar: {:4.2f} ".format(act_linear)+"act: {:5.2f}".format(act.data.numpy()[0])
    plt.plot(x[0],x[1], marker = markers[k], color = colors[k],label=label)
    
plt.legend()
plotmodel(w1, w2, b, X, Y)

plt.show()