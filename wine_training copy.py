import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchsummary import summary
from sklearn.datasets import make_moons
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

features = [0, 9]

wine = datasets.load_wine()
data = wine.data[:, features]
target = wine.target

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

fig, axes = plt.subplots(ncols=2, figsize=(12, 5))

axes[0].scatter(data[:, 0], data[:, 1], c=target, cmap=plt.cm.brg)
axes[0].set_title("Original")
axes[0].set_xlabel(wine.feature_names[features[0]])
axes[0].set_ylabel(wine.feature_names[features[1]])

axes[1].scatter(data_scaled[:, 0], data_scaled[:, 1], c=target, cmap=plt.cm.brg)
axes[1].set_title("Normalizado")
axes[1].set_xlabel(wine.feature_names[features[0]])
axes[1].set_ylabel(wine.feature_names[features[1]])

plt.tight_layout()
plt.show()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU disponível")
else:
    device = torch.device("cpu")
    print("GPU não disponível")

