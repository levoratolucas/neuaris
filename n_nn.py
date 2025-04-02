from torch import nn
import torch

softmax = nn.Softmax()

output = torch.Tensor([2,-1,.5])
outputAtivado = softmax(output)

print(outputAtivado)

test=outputAtivado[0]+outputAtivado[1]+outputAtivado[2]

print(test)