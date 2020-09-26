
"""
Author: Duy-Phuong Dao
Email:  phuongdd.1997@gmail.com (or duyphuongcri@gmail.com)
github: https://github.com/duyphuongcri
"""
from sklearn import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import pandas as pd 
import math
from tqdm import tqdm
import cv2 
class ResNet_block_BN(nn.Module):
    def __init__(self, ch, ksize=3, stride=1, padding=1):
        super(ResNet_block_BN, self).__init__()
        self.resnet = nn.Sequential(
            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ksize, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=ch),
            nn.ReLU(),

            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ksize, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=ch),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.resnet(x) + x

class ResNet_block_no_BN(nn.Module):
    def __init__(self, ch, ksize=3, stride=1, padding=1):
        super(ResNet_block_no_BN, self).__init__()
        self.resnet = nn.Sequential(
            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ksize, stride=stride, padding=padding),
            nn.ReLU(),

            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=ksize, stride=stride, padding=padding),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.resnet(x) + x


class CNN(nn.Module):
    def __init__(self, BN=True):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.maxPool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxPool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxPool3 = nn.MaxPool2d(2, 2)
        if BN:
            self.resnet1 = ResNet_block_BN(ch=16, ksize=3, stride=1, padding=1)
            self.resnet2 = ResNet_block_BN(ch=64, ksize=3, stride=1, padding=1)
            self.resnet3 = ResNet_block_BN(ch=128, ksize=3, stride=1, padding=1)
        else:
            self.resnet1 = ResNet_block_no_BN(ch=16, ksize=3, stride=1, padding=1)
            self.resnet2 = ResNet_block_no_BN(ch=64, ksize=3, stride=1, padding=1)
            self.resnet3 = ResNet_block_no_BN(ch=128, ksize=3, stride=1, padding=1)     

        self.fc1 = nn.Linear(3136, 100, bias=True) #1152
        self.fc2 = nn.Linear(100, 10, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            std = 1/ math.sqrt(weight.size(0))     
            torch.nn.init.uniform_(weight, -std, std)   

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        x1 = self.resnet1(x1)
        x1 = self.maxPool1(x1)

        x2 = self.conv2(x1)
        x2 = self.relu2(x2)
        x2 = self.resnet2(x2)
        x2 = self.maxPool2(x2)

        # x3 = self.conv3(x2)
        # x3 = self.relu3(x3)
        # x3 = self.resnet3(x3)
        # x3 = self.maxPool3(x3)
        x4 = torch.flatten(x2, start_dim=1)
        x5 = self.fc1(x4)
        x6 = self.fc2(x5)
        #out = torch.softmax(x6, dim=1)
        return x6



""" Load data """#######################################
print("Load data...")
num_classes = 10
dft = pd.read_csv('./data/fashion-mnist_train.csv', dtype=int) # read train data
X_train = np.array(dft.drop('label', axis=1))
y_train = np.array(dft['label'])

dft = pd.read_csv('./data/fashion-mnist_test.csv', dtype=int) # read test data
X_test = np.array(dft.drop('label', axis=1))
y_test = np.array(dft['label'])

X_train = torch.Tensor(X_train.reshape((-1, 1, 28, 28)))
y_train = torch.Tensor(y_train)
X_test = torch.Tensor(X_test.reshape((-1, 1, 28, 28)))
y_test = torch.Tensor(y_test)

print("The shape of X train: ", X_train.shape)
print("The shape of y train: ", y_train.shape)
print("The shape of X test:  ", X_test.shape)
print("The shape of y test:  ", y_test.shape)

def load_data(X, y, batch_size):
    n = 0
    num_batch = X.shape[0] // batch_size
    while n < num_batch:
        if n < num_batch - 1:
            X_batch = X[n*batch_size : (n+1)*batch_size]
            y_batch = y[n*batch_size : (n+1)*batch_size]
        else:
            X_batch = X[n*batch_size : ]
            y_batch = y[n*batch_size : ]            
        n += 1
        #print(X_batch.shape, y_batch.shape)
        yield X_batch/255., y_batch


""" Model Settting""" #####################################
lrate = 0.01
batch_size = 256
epochs = 10

np.random.seed(10)
torch.manual_seed(10)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(" GPU is activated" if device else " CPU is activated")
model = CNN(BN=True)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lrate)
criterion = torch.nn.CrossEntropyLoss()
min_acc = 0
if __name__=="__main__":
    for epoch in tqdm(range(epochs)):
        train_loss, test_loss = 0, 0
        model.train()
        for X_batch, y_batch in tqdm(load_data(X_train.to(device), y_train.to(device), batch_size)):
            optimizer.zero_grad()

            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch.long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.shape[0]

        model.eval()
        num_correct = 0
        with torch.no_grad():
            for X_batch, y_batch in load_data(X_test.to(device), y_test.to(device), batch_size):
                y_pred = model(X_batch)
                test_loss += criterion(y_pred, y_batch.long()).item() * X_batch.shape[0]
                argmax = torch.argmax(y_pred, dim=1)
                num_correct += (argmax == y_batch.long()).sum()
 
        accuracy = num_correct.item()*100./10000
        if accuracy > min_acc:
            print("save model")
            min_acc = accuracy
            torch.save(model, "./checkpoint/model_best_BN.pt")
        print("Epoch: {} | Train Loss: {:.5f} | Test Loss: {:.5f} | test Accuracy: {}%".format(epoch+1, 
                                                            train_loss/X_train.shape[0], 
                                                            test_loss/X_test.shape[0],
                                                            accuracy))


    