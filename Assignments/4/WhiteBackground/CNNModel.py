import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.autograd import Variable
import numpy as np
import cv2
import os

from sklearn.metrics import f1_score

# Check if gpu support is available
cuda_avail = torch.cuda.is_available()

class BreakOutDataset(Dataset):
    """BreakOut dataset."""

    def __init__(self, file, transform=None):
        self.images = cv2.imread(file+"_data.png",cv2.IMREAD_GRAYSCALE)
        self.labels = pd.read_csv(file+"_labels.csv")
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_label = np.array(self.labels.iloc[idx,-1])
        image = self.images[idx]
        image = image.reshape(1050,160,3)
#         image = image.reshape(3,1050,160)
        image = np.moveaxis(image, -1, 0)
        image = torch.from_numpy(image).float()  #divided by 255
        img_label = img_label.astype(int)
        img_label = img_label
        img_label = torch.from_numpy(np.eye(2)[img_label]).float()  #One hot encode for BCEloss
        # image = image.reshape(1050,160,3)
        sample = {'image': image, 'label': img_label}
        if self.transform:
            sample = self.transform(sample)
        return sample


# TrainingDataset = BreakOutDataset(file='TrainingData/'+str(1).zfill(8))

# fig = plt.figure()
# for i in range(len(TrainingDataset)):
#     sample = TrainingDataset[i]

#     print(i, sample['image'].shape, sample['label'])

#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     plt.imshow(sample['image'])
#     # show_landmarks(**sample)
#     if i == 3:
#         # plt.show()
#         break

# dataloader = DataLoader(TrainingDataset, batch_size=4,
#                         shuffle=False, num_workers=4)

# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch, sample_batched['image'].size(),sample_batched['label'])


def save_models(epoch):
    torch.save(model.state_dict(), "BreakOutmodel_{}.model".format(epoch))
    print("Chekcpoint saved")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3,out_channels = 32, kernel_size = 3,stride =2,padding =(0,0))
        self.bn2d_1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 32,out_channels = 64, kernel_size = 3,stride =2,padding =(0,0))
        self.bn2d_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride = 2)
        self.fc = nn.Linear(37440,2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.out_layer = nn.Linear(2048, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn2d_1(x)
        
        # print("After Conv1",x)
#         x = self.bn1(x)
        x = self.pool1(x)
        # print("After Max pool1",x)
        x = F.relu(self.conv2(x))
        x = self.bn2d_2(x)
#         x = self.bn2(x)
        # print("After conv 2",x)
        x = self.pool2(x)
        # print("After Max pool 2",x)
        x = x.view(-1, self.num_flat_features(x))
        # print(x)
        x = F.relu(self.fc(x))
        x = self.bn1(x)
        # print("After Hideen Layer",x)
        x = self.out_layer(x)
        # print("After Output layer",x)
        x = F.softmax(x,dim=1)
        # print(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:       # Get the products
            num_features *= s
        return num_features

def accuracy(file):
    accuracy = 0
    total = 0
    TrainingDataset = BreakOutDataset(file=file)
    dataloader = DataLoader(TrainingDataset, batch_size=128, shuffle=False, num_workers=4)
    ls = []
    targets = []
    loss_list = []
    for i_batch, sample_batched in enumerate(dataloader):
        if cuda_avail:
            train_data = (sample_batched['image'].cuda())
            target = (sample_batched['label'].cuda())
        else:                
            train_data = (sample_batched['image'])
            target = (sample_batched['label'])
        # print(train_data.size())
        if(train_data.size()[0] == 1):
            break
        with torch.no_grad():
            pred = net(train_data)
            total +=(len(target))
            # print(len(target))
            loss = criterion(pred, target).cpu()
            maxs, argmax = torch.max(pred, 1)
            _,one_hot_target = torch.max(target,1)
            
            ls = ls + list((argmax.cpu()).numpy())
            targets = targets + list(one_hot_target.cpu().numpy())
            accuracy += torch.sum(torch.eq(argmax,one_hot_target)).float()
            loss_list.append(loss.numpy())
    print("F1 Score", (f1_score(targets, ls, average=None)))
    preds = np.array(ls)
    preds[preds < 0.5] = 0.0
    preds[preds > 0.5] = 1.0
    print("Percentage of 1s in predictions {:4f}".format(np.mean(preds)))
    
    targs = np.array(targets)
    print("Percentage of 1s in targets {:4f}".format(np.mean(targs)))
    print("Validation Loss {:4f}".format(np.mean(loss_list)))
    
    return(accuracy/total)



net = Net().cuda() if cuda_avail else Net()
print("Cuda ",cuda_avail)

# if cuda_avail:
#     class_weights =  torch.FloatTensor([1.0,1.4]).cuda()
# else:
# class_weights =  torch.FloatTensor([1.0, 2.7]).cuda()

# criterion = nn.NLLLoss()
criterion = nn.BCELoss()
# optimizer = optim.SGD(net.parameters(), lr=0.01,momentum = 0.9,weight_decay=1e-5)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, eps=1e-8, amsgrad=False)
epochs = 20
batch_size = 32


for epoch_i in range(epochs):
    running_loss = 0.0
    total_pts = 0
    for i in range(1,501):
        TrainingDataset = BreakOutDataset(file='TrainingData/'+str(i).zfill(8))
        dataloader = DataLoader(TrainingDataset, batch_size=batch_size,shuffle=False, num_workers=4)
        for i_batch, sample_batched in enumerate(dataloader):
            if cuda_avail:
                inputs = Variable(sample_batched['image'].cuda())
                target = Variable(sample_batched['label'].cuda())
            else:                
                inputs = Variable(sample_batched['image'])
                target = Variable(sample_batched['label'])
            optimizer.zero_grad()   # zero the gradient buffers
            if(inputs.size()[0] == 1):
                break
            output = net(inputs)
#             print(output)
#             print(target)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()    # Does the update
            running_loss += loss.cpu().item()
            total_pts += len(target)
            # print(torch.max(output,1)[-1],target)
            # print("probabilties ",output)
            # print(str(loss) +"\n")
#         if i==250:
#             print("Running Loss After ",i,"episodes ",running_loss/total_pts)
        if (i+1)%2 == 0:
            print("Accuracy ",accuracy('TrainingData/'+str(i).zfill(8)))
    print("Runnning Loss on whole DataSet on epoch ",epoch_i," is ",running_loss/total_pts)



# Save the Model
torch.save(net.state_dict(),"BreakOutmodel.model")


# Cross Validation Set
print("Checking Cross Validation Set")
for i in range(151,200):
  TrainingDataset = BreakOutDataset(file='TrainingData/'+str(i).zfill(8))
  dataloader = DataLoader(TrainingDataset, batch_size=batch_size,shuffle=False, num_workers=4)
  print("Accuracy ",accuracy('TrainingData/'+str(i).zfill(8)))


