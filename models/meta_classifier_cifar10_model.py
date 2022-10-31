import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaClassifierCifar10Model(nn.Module):
    def __init__(self):
        super(MetaClassifierCifar10Model, self).__init__()
        #self.gpu = gpu

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.max_pool_0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(64*8*8, 256)
        self.fc = nn.Linear(256, 256)
        self.output = nn.Linear(256, 10)
        self.relu = nn.ReLU()

        '''if gpu:
            self.cuda()'''

    def forward(self, x):
        '''if self.gpu:
            x = x.cuda()'''
        B = x.size()[0]

        x = F.relu(self.conv1(x))
        x = self.max_pool_0(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.max_pool(F.relu(self.conv4(x)))
        x = self.relu(self.linear(x.view(B,64*8*8)))
        x = F.dropout(F.relu(self.fc(x)), 0.5, training=self.training)
        x = self.output(x)

        return x
        
    '''def from_input_to_features(self, x, index):
        B = x.size()[0]

        x = F.relu(self.conv1(x))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.max_pool(F.relu(self.conv4(x)))
        return x
        
    def from_features_to_output(self, x, index):
    
        x = F.relu(self.linear(x.view(-1,64*8*8)))
        x = F.dropout(F.relu(self.fc(x)), 0.5, training=self.training)
        x = self.output(x)
        return x'''
        
        
    def from_input_to_features(self, x, index):
        B = x.size()[0]

        x = F.relu(self.conv1(x))
        x = self.max_pool_0(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = self.max_pool(F.relu(x))
        x = self.relu(self.linear(x.view(-1,64*8*8)))
        
        #print("from_input_to_features:",x.shape)
        return x
        
    def from_features_to_output(self, x, index):
        #x = self.max_pool(F.relu(x))
        #x = F.relu(self.linear(x.view(-1,64*8*8)))
        x = F.dropout(F.relu(self.fc(x)), 0.5, training=self.training)
        x = self.output(x)
        return x
        