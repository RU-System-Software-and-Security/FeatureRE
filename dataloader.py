import torch.utils.data as data
import torch
import torchvision
import torchvision.transforms as transforms
import os
import csv
import random
import numpy as np

from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import Dataset

from io import BytesIO


def get_transform(opt, train=True, pretensor_transform=False):
    add_nad_transform = False
    
    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    if pretensor_transform:
        if train:
            transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop))
            transforms_list.append(transforms.RandomRotation(opt.random_rotation))
            if opt.dataset == "cifar10":
                transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
                
    if add_nad_transform:
        transforms_list.append(transforms.RandomCrop(opt.input_height, padding=4))
        transforms_list.append(transforms.RandomHorizontalFlip())
        

    transforms_list.append(transforms.ToTensor())
    if opt.dataset == "cifar10":
        transforms_list.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
        if add_nad_transform:
            transforms_list.append(Cutout(1,9))
    
    elif opt.dataset == "mnist":
        transforms_list.append(transforms.Normalize([0.1307], [0.3081]))
        if add_nad_transform:
            transforms_list.append(Cutout(1,9))
    elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
        transforms_list.append(transforms.Normalize((0.3403, 0.3121, 0.3214),(0.2724, 0.2608, 0.2669)))
        if add_nad_transform:
            transforms_list.append(Cutout(1,9))
    elif opt.dataset == "imagenet":
        transforms_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        if add_nad_transform:
            transforms_list.append(Cutout(1,9))
    else:
        raise Exception("Invalid Dataset")

    return transforms.Compose(transforms_list)



class GTSRB(data.Dataset):
    def __init__(self, opt, train, transforms):
        super(GTSRB, self).__init__()
        if train:
            self.data_folder = os.path.join(opt.data_root, "GTSRB/Train")
            self.images, self.labels = self._get_data_train_list()
        else:
            self.data_folder = os.path.join(opt.data_root, "GTSRB/Test")
            self.images, self.labels = self._get_data_test_list()

        self.transforms = transforms

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                images.append(prefix + row[0])
                labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            images.append(self.data_folder + "/" + row[0])
            labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        return image, label

def get_dataloader_partial_split(opt, train_fraction=0.1, train=True, pretensor_transform=False,shuffle=True,return_index = False):
    data_fraction = train_fraction
    
    transform_train = get_transform(opt, True, pretensor_transform)
    transform_test = get_transform(opt, False, pretensor_transform)
    
    transform = transform_train
    
    if opt.dataset == "gtsrb":
        dataset = GTSRB(opt, train, transform_train)
        dataset_test = GTSRB(opt, train, transform_test)
        class_num=43
    elif opt.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(opt.data_root, train, transform=transform_train, download=True)
        dataset_test = torchvision.datasets.MNIST(opt.data_root, train, transform=transform_test, download=True)

        class_num=10
    elif opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform=transform_train, download=True)
        dataset_test = torchvision.datasets.CIFAR10(opt.data_root, train, transform=transform_test, download=True)
        class_num=10
    elif opt.dataset == "celeba":
        if train:
            split = "train"
        else:
            split = "test"
        dataset = CelebA_attr(opt, split, transform)
        class_num=8
    elif opt.dataset == "imagenet":
        if train==True:
            file_dir = "/workspace/data/imagenet/train"
        elif train==False:
            file_dir = "/workspace/data/imagenet/val"
        dataset = torchvision.datasets.ImageFolder(
            file_dir,
            transform
            )
        dataset_test = torchvision.datasets.ImageFolder(
            file_dir,
            transform
            )
        class_num=1000
    else:
        raise Exception("Invalid dataset")
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=True)
    #finetuneset = torch.utils.data.Subset(dataset, range(0,dataset.__len__(),int(1/data_fraction)))
    dataloader_total = torch.utils.data.DataLoader(dataset, batch_size=1, pin_memory=True,num_workers=opt.num_workers, shuffle=False)
    
    idx = []
    counter = [0]*class_num
    for batch_idx, (inputs, targets) in enumerate(dataloader_total):

        if counter[targets.item()]<int(dataset.__len__()*data_fraction/class_num):
            idx.append(batch_idx)
            counter[targets.item()] = counter[targets.item()] + 1

    del dataloader_total
    trainset = torch.utils.data.Subset(dataset,idx)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs,pin_memory=True, num_workers=opt.num_workers, shuffle=shuffle)

    test_idx = list(set(range(dataset.__len__())) - set(idx))
    testset = torch.utils.data.Subset(dataset_test,test_idx)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.bs,pin_memory=True, num_workers=opt.num_workers, shuffle=shuffle)

    if return_index:
        return trainset, transform, trainloader, testset, testloader,idx,test_idx
    else:
        return trainset, transform, trainloader, testset, testloader

def get_dataloader_label_partial(opt, dataset_total, label=0):

    dataloader_total = torch.utils.data.DataLoader(dataset_total, batch_size=1,pin_memory=True,num_workers=opt.num_workers, shuffle=False)
    idx = []
    for batch_idx, (inputs, targets) in enumerate(dataloader_total):
        if targets.item() == label:
            idx.append(batch_idx)
    del dataloader_total
    class_dataset = torch.utils.data.Subset(dataset_total,idx)
    dataloader_class = torch.utils.data.DataLoader(class_dataset, batch_size=opt.bs,pin_memory=True,num_workers=opt.num_workers, shuffle=True)

    return dataloader_class

def get_dataloader_label_remove(opt, dataset_total, label=0, idx=None):

    dataloader_total = torch.utils.data.DataLoader(dataset_total, batch_size=1,pin_memory=True,num_workers=opt.num_workers, shuffle=False)
    if idx is None:
        idx = []
        for batch_idx, (inputs, targets) in enumerate(dataloader_total):
            #_input, _label = self.model.get_data(data)
            #_input = _input.view(1, _input.shape[0], _input.shape[1], _input.shape[2])
            if targets.item() != label:
                idx.append(batch_idx)

    del dataloader_total
    class_dataset = torch.utils.data.Subset(dataset_total,idx)
    dataloader_class = torch.utils.data.DataLoader(class_dataset, batch_size=opt.bs,pin_memory=True,num_workers=opt.num_workers, shuffle=True)

    return dataloader_class

def main():
    pass


if __name__ == "__main__":
    main()
