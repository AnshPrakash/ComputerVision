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

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

cuda_avail = torch.cuda.is_available()


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
        # x = self.bn1(x)
        x = self.pool1(x)
        # print("After Max pool1",x)
        x = F.relu(self.conv2(x))
        x = self.bn2d_2(x)
        # x = self.bn2(x)
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

device = torch.device('cpu')
model_file = "BreakOutmodel.model"
net = Net()
net.load_state_dict(torch.load(model_file,map_location=device))
net.eval()

# print(list(net.parameters()))

code_dir = os.getcwd()

test_dir = "test_dataset"
# valid_set_dir = "./validation_dataset"
os.chdir(test_dir)
curr_dir = os.getcwd()
# with open("rewards.csv") as f:
# 	rewards = pd.read_csv(f,header = None)

# labels = np.array(rewards.iloc[:,-1])


def imageStacked():
	imglist = [str(i)+".png" for i in range(5)]
	img_array = [cv2.imread(x) for x in imglist]
	temp = img_array[0]
	temp = temp
	for img in img_array[1:]:
		temp = np.vstack((temp,img))
	# temp = temp.ravel()
	return(temp)



folders = list(filter(os.path.isdir, os.listdir('.')))
folders.sort()

test  = []
labels = [0]*len(folders)

predictions = []
for folder in (folders):
    os.chdir(folder)
    print(folder)
    img = imageStacked()
    img = np.moveaxis(img,-1,0)
    img = img.reshape(1,3,1050,160)
    img = torch.from_numpy(img).float()
    if cuda_avail:
        inputs = Variable(img.cuda())
    else:                
        inputs = Variable(img)
    with torch.no_grad():
        pred = net(inputs)
        _, argmax = torch.max(pred, 1)
        predictions = predictions + (argmax.tolist())
        os.chdir(curr_dir)
predictions =np.array(predictions)
print(predictions)

os.chdir(code_dir)
df = pd.DataFrame(predictions,columns=['Prediction'])
df.Prediction = df.Prediction.astype(int)
df.index.name = 'id'
df.to_csv('CNN_test.csv')

print("CNN Done")


# conf = (confusion_matrix(labels,predictions,labels=[0,1]))
# print(conf)
# f1 = f1_score(labels, predictions, average=None)
# print("F1_Score \n",f1)
# print("f1_score_micro \n",f1_score(labels,predictions, average='micro'))

