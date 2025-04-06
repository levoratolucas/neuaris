import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchsummary import summary
from sklearn.datasets import make_moons
from sklearn import datasets



wine = datasets.load_wine()
data = wine.data
target = wine.target
print(data.shape)   
print(target.shape)
print(wine.feature_names,wine.target_names)


class WineClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(WineClassifier, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax()
        
    def forward(self, X):
        feature = self.relu(self.hidden(X))
        output = self.softmax(self.output(feature))
        return output
    
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("disponivel")
else:
    device = torch.device("cpu")
    print("neagtivo")
    
input_size = data.shape[1]
hidden_size = 32
output_size = len(wine.target_names)
        
net = WineClassifier(input_size, hidden_size, output_size).to(device)