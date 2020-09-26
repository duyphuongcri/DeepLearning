"""
Author: Duy-Phuong Dao
Email: phuongdd.1997@gmail.com (or duyphuongcri@gmail.com)
github: https://github.com/duyphuongcri
"""

import numpy as np
import torch 
import torch.nn as nn
import math
class XOR_problem(nn. Module):
    def __init__(self, num_features, num_out):
        super(XOR_problem, self).__init__()
        self.neurals = nn.Sequential(
            nn.Linear(num_features, 16, bias=True),
            nn.ReLU(),
            nn.Linear(16, num_out, bias=True),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            stdv = 1.0 / math.sqrt(weight.size(0))
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, X):

        out = torch.sigmoid(self.neurals(X))
        return out

############ DATA ######################
X = torch.Tensor([[0, 0],
                [1, 1],
                [0, 1],
                [1, 0]])
Y = torch.Tensor([0, 0, 1, 1])
Y = torch.reshape(Y, (4,1))
print("X shape: ", X.shape)
print("Y shape: ", Y.shape)
################
model = XOR_problem(num_features=2, num_out=1)
optimizer = torch.optim.SGD(model.parameters(), lr=1)
def MAE(pred, true):
    return abs(true - pred).sum()/pred.shape[0]

## train -----------------------------------------------
for epoch in range(50000):
    optimizer.zero_grad()

    pred = model(X)
    # measure loss
    loss = MAE(pred, Y)
    # optimize model
    loss.backward()
    optimizer.step()
    #print("Epoch: {} | Loss: {}".format(epoch, loss.item()))

## test ------------------------------------------------
pred = model(X)
print(pred)
