import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchsummary import summary
from sklearn.datasets import make_moons
from sklearn import datasets



diabetes = datasets.load_diabetes()
data = diabetes.data
target = diabetes.target
print(data.shape)
print(target.shape)
print(diabetes.feature_names,diabetes.target_names)