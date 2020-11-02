import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import sklearn.metrics as metrics
 
from models import DeepLocNoAtt
from tqdm import trange

#Hyperparameters
batch_size = 128
num_epochs = 120

#Data 
train = np.load('data/train.npz')
test = np.load('data/test.npz')

trainset = data.TensorDataset(torch.Tensor(train['X_train']), torch.Tensor(train['mask_train']), torch.Tensor(train['y_train']))
trainloader = data.DataLoader(trainset, batch_size=batch_size)

testset = data.TensorDataset(torch.Tensor(test['X_test']), torch.Tensor(test['mask_test']), torch.Tensor(test['y_test']))
testloader = data.DataLoader(testset, batch_size=batch_size)
print("Data loaded")

#Training                               
net = DeepLocNoAtt()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print("Model initialized")

min_loss = float('inf')
PATH = './saved_params/cnn_bilstm.pth'
print("Training")

t = trange(num_epochs)
for epoch in t:
    losses, accs = [], []
    for i, data in enumerate(trainloader, 0):
        inputs, masks, labels = data
        labels = labels.long()
        optimizer.zero_grad()
        # print(inputs.size())
        outputs = net.forward(inputs, masks)
        # print(outputs.size(), labels.size())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if loss < min_loss:
            torch.save(net.state_dict(), PATH)
        losses.append(loss)
        correct = labels.eq(outputs.argmax(-1)).sum()
        accs.append(int(correct)/batch_size)
    t.set_postfix(loss=sum(losses)/len(losses), acc=(sum(accs)/len(accs)))

print("Results")
#Testing
vlosses, vaccs = [], []
for i, data in enumerate(testloader, 0):
    inputs, masks, labels = data
    outputs = net.forward(inputs, masks)
    loss = criterion(outputs, labels.long())
    vlosses.append(loss)
    correct = labels.eq(outputs.argmax(-1)).sum()
    vaccs.append(int(correct)/batch_size)

avg_loss = sum(vlosses)/len(vlosses)
avg_acc = sum(vaccs)/len(vaccs)
print('avg loss: {}\navg acc: {}'.format(avg_loss, avg_acc))