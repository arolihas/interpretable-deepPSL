import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import sklearn.metrics as metrics
 
from models import DeepLoc

#Hyperparameters
batch_size = 128
num_epochs = 10

#Data 
train = np.load('data/train.npz')
test = np.load('data/test.npz')

trainset = data.TensorDataset(torch.Tensor(train['X_train']), torch.Tensor(train['mask_train']), torch.Tensor(train['y_train']))
trainloader = data.DataLoader(trainset, batch_size=batch_size)

testset = data.TensorDataset(torch.Tensor(test['X_test']), torch.Tensor(test['mask_test']), torch.Tensor(test['y_test']))
testloader = data.DataLoader(testset, batch_size=batch_size)
print("Data loaded")

#Training                               
net = DeepLoc()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print("Model initialized")

min_loss = float('inf')
PATH = './saved_params/cnn_bilstm_attn.pth'
print("Training")
for epoch in range(num_epochs):
    losses, accs = [], []
    for i, data in enumerate(trainloader, 0):
        inputs, masks, labels = data
        labels = labels.long()
        optimizer.zero_grad()
        outputs = net.forward(inputs, masks)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if loss < min_loss:
            torch.save(net.state_dict(), PATH)
        losses.append(loss)
        correct = labels.eq(outputs.argmax(-1)).sum()
        accs.append(int(correct)/batch_size)
    print("Epoch {} \nLoss {}\nAccuracy {}".format(epoch, sum(losses)/len(losses), sum(accs)/len(accs)))

print("Results")
#Testing
losses, accs = [], []
for i, data in enumerate(testloader, 0):
    inputs, masks, labels = data
    outputs = net.forward(inputs, masks)
    loss = criterion(outputs, labels.long())
    losses.append(loss)
    correct = labels.eq(outputs.argmax(-1)).sum()
    accs.append(int(correct)/batch_size)

avg_loss = sum(losses)/len(losses)
avg_acc = sum(accs)/len(accs)
print('avg loss: {}\navg acc: {}'.format(avg_loss, avg_acc))