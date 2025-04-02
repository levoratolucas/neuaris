import torch
import torch.nn as nn
import torch.nn.functional as F


import matplotlib.pyplot as plt
from torch import nn
from torchsummary import summary
from sklearn.datasets import make_moons

X1, Y1 = make_moons(n_samples=300, noise=0.2)

plt.scatter(X1[:,0], X1[:, 1], marker='o', c=Y1, s=25, edgecolors='k')

plt.show()



input_size = 2
hidden_size = 8
output_size = 1

# net = nn.Sequential(
#     nn.Linear(input_size,hidden_size),# escondida
#     nn.ReLU(),#Ativação não linear
#     nn.Linear(hidden_size,output_size)#output (saida)
# )

# pred = net(tensorX1)

# print(summary(net,input_size=(1,input_size)))


class  minhaNn(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(minhaNn , self).__init__()
        
        # definir arquitetura
        self.hidden = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size,output_size)
        
    def forward(self,X):
        
        
        # gerar saida
        hidden = self.relu(self.hidden(X))
        output = self.output(hidden)
        return output
    
    
net = minhaNn(input_size,hidden_size,output_size)


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("disponivel")
else:
    device = torch.device("cpu")
    print("neagtivo")
    
net.to(device)
tensorX1 = torch.from_numpy(X1).float()
tensorX1 = tensorX1.to(device)