import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
from sklearn.datasets import make_moons
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


features = [0,9]

wine = datasets.load_wine()
data = wine.data[:,features]

scaler = StandardScaler()
data = scaler.fit_transform(data)

target = wine.target

plt.scatter(data[:,0],data[:,1],cmap=plt.cm.brg,c=target)
plt.xlabel(wine.feature_names[features[0]])
plt.ylabel(wine.feature_names[features[1]])

plt.show()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU disponível")
else:
    device = torch.device("cpu")
    print("GPU não disponível")


input_size = data.shape[1]          # NUMERO DE CARACTERISTICAS NO CASO DUAS, ESTÁ NA LINHA 10 DELIMITADOS NA FEATURES
hidden_size = 32                    # CAMADA OCULTA DA ESCOLHA DO DEV
output_size = len(wine.target_names)# AS CLASSES DE SAIDA EXEMPLO CACHORRO GATO CORUJA



def plot_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    spacing = min(x_max - x_min, y_max - y_min) / 100

    XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing),
                         np.arange(y_min, y_max, spacing))

    data = np.hstack((XX.ravel().reshape(-1, 1),
                      YY.ravel().reshape(-1, 1)))

    # # For binary problems
    # db_prob = model(Variable(torch.Tensor(data).cuda()))
    # clf = np.where(db_prob.cpu().data < 0.5, 0, 1)

    # For multi-class problems
    db_prob = model(torch.Tensor(data).to(device))
    clf = np.argmax(db_prob.cpu().data.numpy(), axis=-1)

    Z = clf.reshape(XX.shape)

    plt.contourf(XX, YY, Z, cmap=plt.cm.brg, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.brg, edgecolors='k', s=25)
