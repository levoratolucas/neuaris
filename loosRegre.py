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
# print(diabetes.feature_names,diabetes.target_names)

class DiabetesClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(DiabetesClassifier, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

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
output_size = 1

net = DiabetesClassifier(input_size, hidden_size, output_size).to(device)

criterion = nn.MSELoss().to(device)

Xtns = torch.from_numpy(data).float().to(device)
Ytns = torch.from_numpy(target).float().to(device)


print(Xtns.shape)
print(Ytns.shape)

pred = net(Xtns)
print(pred.shape)
loss = criterion(pred.squeeze(), Ytns)
print(loss)