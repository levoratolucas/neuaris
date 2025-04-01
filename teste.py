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
tnsr[0,2] = -11

# print(tnsr)
# tensor([[  0.0124,  -1.6754, -11.0000],
#         [ -0.3589,   0.4211,   0.4958],
#         [ -0.4231,  -1.5521,  -1.5596]])

#________________________________________________________

#____________________ indexando __________

print(tnsr[0:2])
# tensor([[ -0.7572,   1.0013, -11.0000],
#         [  0.7777,   0.3247,  -1.0851]])
#______________________________________________________

################ convertendo pra numpy{
# arr = tnsr.data.numpy()

# print(arr)


# [[-0.3567002   0.9054505  -0.0129268 ]
#  [ 0.04520945  1.542316   -0.317889  ]
#  [-0.23926812  1.5377839   0.70624363]]

###################### }
# ____________________________________________________________________
