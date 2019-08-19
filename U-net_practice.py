import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from utils import load_data,reshape_dimension 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding= 1)
        self.conv1_x = nn.Conv2d(8, 8, 3, padding= 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding = 1)
        self.conv2_x = nn.Conv2d(16, 16, 3, padding= 1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding = 1)
        self.conv3_x = nn.Conv2d(32, 32, 3, padding= 1)
        self.conv4 = nn.Conv2d(32, 64, 3, padding = 1)
        self.conv4_x = nn.Conv2d(64, 64, 3, padding= 1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv5_x = nn.Conv2d(128, 128, 3, padding= 1)
        self.convT1 = nn.ConvTranspose2d(128, 64, 3, padding = 1,stride=2, output_padding=1)
        self.conv6 =  nn.Conv2d(128, 64, 3, padding= 1)
        self.conv6_x =  nn.Conv2d(64, 64, 3, padding= 1)
        self.convT2 = nn.ConvTranspose2d(64, 32, 3, padding = 1,stride=2, output_padding=1)
        self.conv7 =  nn.Conv2d(64, 32, 3, padding= 1)
        self.conv7_x =  nn.Conv2d(32, 32, 3, padding= 1)
        self.convT3 = nn.ConvTranspose2d(32, 16, 3, padding = 1,stride=2, output_padding=1)
        self.conv8 =  nn.Conv2d(32, 16, 3, padding= 1)
        self.conv8_x =  nn.Conv2d(16, 16, 3, padding= 1)
        self.convT4 = nn.ConvTranspose2d(16, 8, 3,padding = 1,stride=2, output_padding=1)
        self.conv9 =  nn.Conv2d(16, 8, 3, padding= 1)
        self.conv9_x =  nn.Conv2d(8, 8, 3, padding= 1)
        self.conv10 = nn.Conv2d(8, 4, 1)

    def forward(self, x):
        con1_1 = F.relu(self.conv1(x))
        con1_2 = F.relu(self.conv1_x(con1_1))
        pool_1 = self.pool(con1_2)
        con2_1 = F.relu(self.conv2(pool_1))
        con2_2 = F.relu(self.conv2_x(con2_1))
        pool_2 = self.pool(con2_2)
        con3_1 = F.relu(self.conv3(pool_2))
        con3_2 = F.relu(self.conv3_x(con3_1))
        pool_3 = self.pool(con3_2)
        con4_1 = F.relu(self.conv4(pool_3))
        con4_2 = F.relu(self.conv4_x(con4_1))
        pool_4 = self.pool(con4_2)
        con5_1 = F.relu(self.conv5(pool_4))
        con5_2 = F.relu(self.conv5_x(con5_1))
        up_1_1 = self.convT1(con5_2)
        up_1_2 = torch.cat([con4_2,up_1_1],dim=1)
        con6_1 = F.relu(self.conv6(up_1_2))
        con6_2 = F.relu(self.conv6_x(con6_1))
        up_2_1 = self.convT2(con6_2)
        up_2_2 = torch.cat([con3_2,up_2_1],dim=1)
        con7_1 = F.relu(self.conv7(up_2_2))
        con7_2 = F.relu(self.conv7_x(con7_1))
        up_3_1 = self.convT3(con7_2)
        up_3_2 = torch.cat([con2_2,up_3_1],dim=1)
        con8_1 = F.relu(self.conv8(up_3_2))
        con8_2 = F.relu(self.conv8_x(con8_1))
        up_4_1 = self.convT4(con8_2)
        up_4_2 = torch.cat([con1_2,up_4_1],dim=1)
        con9_1 = F.relu(self.conv9(up_4_2))
        con9_2 = F.relu(self.conv9_x(con9_1))
        con10 = self.conv10(con9_2)
        return con10


class Net_test(nn.Module):
    def __init__(self):
        super(Net_test, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        print(x.shape)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
def mini_main():
    net = Net_test()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    x_data = torch.randn(30,3,32,32)
    y_data = torch.randn(30,10)
    data = [x_data,y_data]
    for epoch in range(2):  # loop over the dataset multiple times
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss = loss.item()
        print(running_loss)
    print('Finished Training')

def shuffle_list(num,batch):
    num = num//batch*batch
    
    lister =np.arange(num)
    np.random.shuffle(lister)
    lister = lister.reshape(-1,batch)
    print(lister)

    return lister

def cross_criterion(outputs, labels):
    # for batches in zip(outputs,labels):
    #     outputs
    return np.sum(-labels*np.log(outputs))

def main():
    axis = 2
    batch = 16
    epoch = 10
    model_name = 'z_model_torch.h5'
    # height,width,channel = 256,256,1
    X_train, Y_train = load_data(group=1)
    X_train, Y_train = reshape_dimension(X_train, Y_train,axis,zero_remove=True)
    X_train, Y_train = np.transpose(X_train, (0,3, 1, 2)),np.transpose(Y_train,(0,3,1,2))
    print(X_train.shape)
    X_train,Y_train = torch.tensor(X_train.astype('float32')),torch.tensor(Y_train.astype('long'))
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    lister = shuffle_list(len(X_train),batch)

    net = Net()
    
    # criterion = cross_criterion()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    print('1')
    for epoches in range(epoch):  # loop over the dataset multiple times
        print(epoches)
        running_loss = 0.0
        for i, lists in enumerate(lister):
            print('2')
            # get the inputs
            inputs, labels = X_train[lists,:,:,:],Y_train[lists,:,:,:]
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            print('3')
            # forward + backward + optimize
            outputs = net(inputs)
            loss = cross_criterion(outputs, labels)
            loss.backward()
            print('4')
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                
    print('Finished Training')


if __name__ == '__main__':
    mini_main()