import torch
from torch import nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


torch.manual_seed(42)

percep = nn.Linear(in_features=3, out_features=1)


# for nome, tensor in percep.named_parameters():
#     print(nome, tensor.data)

# print(percep.weight.data)
# print(percep.bias.data)


# w1 * x1 + w2*x2 +w3*x3 + b = 0

def plot3d(percep):

    w1, w2, w3 = percep.weight.data.numpy()[0]
    b = percep.bias.data.numpy()
    X1 = np.linspace(-1, 1, 10)
    X2 = np.linspace(-1, 1, 10)

    X1, X2 = np.meshgrid(X1, X2)
    X3 = (b - w1*X1 - w2*X2) / w3

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.view_init(azim = 180)
    ax.plot_surface(X1,X2,X3, cmap= 'plasma')
    
# plot3d(percep)

X = torch.Tensor([0,1,2])
y = percep(X)

print(y)
plot3d(percep)
plt.plot([X[0]],[X[1]],[X[2]],marker = 'o', color ='r' )

plt.show()

