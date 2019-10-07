## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.conv4 = nn.Conv2d(128, 256, 5)
        
        
        # Define pooling layer
        self.max_pool_1 = nn.MaxPool2d(3,2)
        self.max_pool_2 = nn.MaxPool2d(4,3)
        self.max_pool_3 = nn.MaxPool2d(3,3)
        self.max_pool_4 = nn.MaxPool2d(6,3)
        # Define Batch Normalization
        self.batch_norm_1 = nn.BatchNorm2d(32)
        self.batch_norm_2 = nn.BatchNorm2d(64)
        self.batch_norm_3 = nn.BatchNorm2d(128)
        self.batch_norm_4 = nn.BatchNorm2d(256)
        # Define dropout function
        self.dropout = nn.Dropout(0.2)
        # FC layer
        self.fc1 = nn.Linear(256*1*1, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        # Weight initialization
        I.xavier_uniform(self.fc1.weight.data)
        I.xavier_uniform(self.fc2.weight.data)
        I.xavier_uniform(self.fc3.weight.data)
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.batch_norm_1(self.max_pool_2(self.conv1(x)))) # (224 - 5) / 1 + 1 = 220.. (220 - 4) / 3 + 1 = 73
        x = F.relu(self.batch_norm_2(self.max_pool_1(self.conv2(x)))) # (73 - 5) / 1 + 1 = 69.. (69 - 3) / 2 + 1 = 34
        x = F.relu(self.batch_norm_3(self.max_pool_3(self.conv3(x)))) # (34 - 5) / 1 + 1 = 30.. (30 - 3) / 3 + 1 = 10
        x = F.relu(self.batch_norm_4(self.max_pool_4(self.conv4(x)))) # (10 - 5) / 1 + 1 = 6.. (6 - 6) / 3 + 1 = 1
        # Flatten the image
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
