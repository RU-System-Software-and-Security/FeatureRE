# This part is borrowed from https://github.com/huawei-noah/Data-Efficient-Model-Compression

import torch.nn as nn


class LeNet5(nn.Module):

    def __init__(self,in_channels=1):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=(5, 5))
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(84, 10)

    def forward(self, img, out_feature=False):
        output = self.conv1(img)
        output = self.relu1(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.maxpool2(output)
        output = self.conv3(output)
        output = self.relu3(output)
        feature = output.view(-1, 120)
        output = self.fc1(feature)
        output = self.relu4(output)
        output = self.fc2(output)
        if out_feature == False:
            return output
        else:
            return output,feature
            
    def forward_activations(self, x):
        
        inner_output_list = []
        output = self.conv1(x)
        inner_output_list.append(output)
        output = self.relu1(output)
        
        output = self.maxpool1(output)
        output = self.conv2(output)
        inner_output_list.append(output)
        output = self.relu2(output)
        
        output = self.maxpool2(output)
        output = self.conv3(output)
        inner_output_list.append(output)
        output = self.relu3(output)
        
        feature = output.view(-1, 120)
        output = self.fc1(feature)
        inner_output_list.append(output)
        output = self.relu4(output)
        out = self.fc2(output)

        return inner_output_list[0],inner_output_list[1],inner_output_list[2],inner_output_list[3],out
            
    def get_all_inner_activation(self, img):
        inner_output_list = []
        output = self.conv1(img)
        inner_output_list.append(output)
        output = self.relu1(output)
        
        output = self.maxpool1(output)
        output = self.conv2(output)
        inner_output_list.append(output)
        output = self.relu2(output)
        
        output = self.maxpool2(output)
        output = self.conv3(output)
        inner_output_list.append(output)
        output = self.relu3(output)
        
        feature = output.view(-1, 120)
        output = self.fc1(feature)
        inner_output_list.append(output)
        output = self.relu4(output)
        output = self.fc2(output)

        return output,inner_output_list
        
    def from_input_to_features(self, x, index):
        output = self.conv1(x)
        output = self.relu1(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.maxpool2(output)
        output = self.conv3(output)
        output = self.relu3(output)
        feature = output.view(-1, 120)
        output = self.fc1(feature)
        x = self.relu4(output)
        return x
        
    def from_features_to_output(self, x, index):
        x = self.fc2(x)
        return x
    

class LeNet5Half(nn.Module):

    def __init__(self):
        super(LeNet5Half, self).__init__()

        self.conv1 = nn.Conv2d(1, 3, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(3, 8, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(8, 60, kernel_size=(5, 5))
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(60, 42)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(42, 10)

    def forward(self, img, out_feature=False):
        output = self.conv1(img)
        output = self.relu1(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.maxpool2(output)
        output = self.conv3(output)
        output = self.relu3(output)
        feature = output.view(-1, 60)
        output = self.fc1(feature)
        output = self.relu4(output)
        output = self.fc2(output)
        if out_feature == False:
            return output
        else:
            return output,feature