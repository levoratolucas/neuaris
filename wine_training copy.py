import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import os
from moviepy import ImageSequenceClip
from io import BytesIO
from PIL import Image

# Escolhendo duas features para visualização
features = [0, 9]
wine = datasets.load_wine()
data = wine.data[:, features]

scaler = StandardScaler()
data = scaler.fit_transform(data)

target = wine.target

# Verificando uso da GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

# Configurando rede neural
input_size = data.shape[1]
hidden_size = 32
output_size = len(wine.target_names)

net = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size),
    nn.Softmax(dim=-1)
).to(device)

# Função para desenhar os limites de decisão
def gerar_frame(X, y, model):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    spacing = min(x_max - x_min, y_max - y_min) / 100

    XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing),
                         np.arange(y_min, y_max, spacing))

    data_plot = np.hstack((XX.ravel().reshape(-1, 1),
                           YY.ravel().reshape(-1, 1)))

    db_prob = model(torch.Tensor(data_plot).to(device))
    clf = np.argmax(db_prob.cpu().data.numpy(), axis=-1)
    Z = clf.reshape(XX.shape)

    plt.figure()
    plt.contourf(XX, YY, Z, cmap=plt.cm.brg, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.brg, edgecolors='k', s=25)

    # Salvar o frame em memória
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return Image.open(buf)

# Função de perda e otimizador
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=1e-2)

X = torch.Tensor(data).to(device)
y = torch.LongTensor(target).to(device)

def firstOptmizer(X, y):
    optimizer.zero_grad()
    pred = net(X)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()

# Lista de frames
frames = []

# Loop de treinamento
for i in range(1000):
    firstOptmizer(X, y)

    if i % 20 == 0:
        frame = gerar_frame(data, target, net)
        frames.append(frame)

# Criar vídeo com MoviePy
clip = ImageSequenceClip([np.array(f) for f in frames], fps=5)
clip.write_videofile("treinamento.mp4", codec="libx264")

print("✅ Vídeo salvo como treinamento.mp4")
