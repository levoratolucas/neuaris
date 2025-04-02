import torch
import numpy as np

# lista = [[1, 2, 3],
#          [4, 5, 6]]

# tns = torch.Tensor(lista)
# tns = torch.FloatTensor(lista) # Float
# tns = torch.DoubleTensor(lista) #Double
# tns = torch.LongTensor(lista) # inteiros

# print(tns.dtype)


#________________________ criando e convertendo  com numpy _______________-
# arr = np.random.rand(3,4)


# tns_numpy = torch.from_numpy(arr)

# print(arr)
# print(tns_numpy)


# [[0.36669568 0.21454591 0.65955331 0.08299758]
#  [0.7856933  0.87203522 0.28290128 0.98711187]
#  [0.02968316 0.17724603 0.34023639 0.85334299]]
# tensor([[0.3667, 0.2145, 0.6596, 0.0830],
#         [0.7857, 0.8720, 0.2829, 0.9871],
#         [0.0297, 0.1772, 0.3402, 0.8533]], dtype=torch.float64)

# ____________________________________________________________________
#_________________________________________ gerando com torch__________

tns1 = torch.ones(2,3)
tns0 = torch.zeros(4,5)
tnsr = torch.randn(3,3)




# tns = tnsr[0:2, 0:3] ## redimencionando pra concatenar
# tns_out = torch.cat( (tns1, tns), dim=0 ) # concatena tensores de um mesmo tamanho
# print(tns_out)

# print(tns1)
# print(tns0)
# print(tnsr)

# tensor([[1., 1., 1.],
#         [1., 1., 1.]])
# tensor([[0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.]])
# tensor([[-0.1128, -0.8623,  0.9200],
#         [-0.4249, -1.0249,  2.1400],
#         [ 0.9468, -0.4204, -0.8866]])

# ______________alterando tensor ______________________
# tnsr[0,2] = -11

# print(tnsr)
# tensor([[  0.0124,  -1.6754, -11.0000],
#         [ -0.3589,   0.4211,   0.4958],
#         [ -0.4231,  -1.5521,  -1.5596]])

#________________________________________________________

#____________________ indexando __________

# print(tnsr[0:2])
# tensor([[ -0.7572,   1.0013, -11.0000],
#         [  0.7777,   0.3247,  -1.0851]])


# print(tnsr[:,2])

# tensor([-11.0000,  -0.4223,  -0.8959])

# print(tnsr[0,2])
# tensor(-11.)
#______________________________________________________

################ convertendo pra numpy{
# arr = tnsr.data.numpy()

# print(arr)


# [[-0.3567002   0.9054505  -0.0129268 ]
#  [ 0.04520945  1.542316   -0.317889  ]
#  [-0.23926812  1.5377839   0.70624363]]

###################### }
# ____________________________________________________________________


##################### operações com tensores de tamanhos diferentes



# print(tnsr.shape)
# print(tns1.shape)

# torch.Size([3, 3])
# torch.Size([2, 3])

# tns = tnsr[0:2, :]


# print(tns)

# tensor([[  0.5960,  -1.3867, -11.0000],
#         [ -0.1597,  -1.2106,   0.5627]])


# print(tns.shape)
# print(tns1.shape)

# torch.Size([2, 3])
# torch.Size([2, 3])

# print(tns,"\n",tns1,"\n",tns + tns1)

# tensor([[ -0.6688,  -0.7944, -11.0000],
#         [  3.0286,   0.4787,  -0.5803]]) 
#  tensor([[1., 1., 1.],
#         [1., 1., 1.]])
#  tensor([[  0.3312,   0.2056, -10.0000],
#         [  4.0286,   1.4787,   0.4197]])


##########################################


# redimencionando tensores

# tns = torch.randn(2,2,3)

# print(tns)

# tensor([[[-0.7487,  1.1941, -0.4559],[ 0.4920,  0.8041, -1.4544]],
#         [[-0.1889, -0.8754,  0.9833],[ 1.3394,  0.4039, -0.8883]]])

# tns = tns.view(12)
# tns = tns.view(-1)

# print(tns)


# # tensor([ 0.2006,  0.4789,  1.7802,  0.5436, -0.5189,  1.8464, -0.9948, -0.1248,
# #         -1.4702,  0.1530, -0.0818, -0.1528])

# print(tns.shape)
# torch.Size([12])


# tns = tns.view(4,3)

# print(tns)

# tensor([[-0.2235,  0.7202,  0.5922],
#         [ 0.4767, -0.3618,  0.6645],
#         [ 0.1318,  1.2619,  0.6761],
#         [ 0.6894,  0.0263, -0.9461]])

# print(tns.shape)

# torch.Size([4, 3])

################# mantendo a primeira dimensão e achatando o restantes


# tns = tns.view(tns.size(0),-1)

# print(tns)


# # tensor([[-0.4449, -0.2939, -0.2247,  0.2039, -0.6229,  0.7011],
# #         [ 0.0941,  0.5444,  1.0702,  0.1484,  1.9551, -1.0597]])

# print(tns.shape)

# # torch.Size([2, 6])

# ##############################

# # cast na gpu

# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print("disponivel")
# else:
#     device = torch.device("cpu")
#     print("neagtivo")
    
# # caso tenha gpu fica disponivel caso não, use colab

# tns = torch.randn(10)
# tns = tns.to(device)
# print(tns)

# ######## no colab 
# # tensor([ 1.1864, -1.6370,  0.5734,  1.3719,  1.5507, -1.0945,  0.4908,  0.5354,
# #         -0.2227,  2.5599], device='cuda:0')

# ######## no meu not
# # tensor([-1.3771, -1.8955, -1.8313, -0.6297, -0.4443,  0.0027,  0.0844, -1.6817,
# #          0.2080,  0.9132])