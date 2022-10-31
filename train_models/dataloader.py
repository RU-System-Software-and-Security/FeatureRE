import torch.utils.data as data
import torch
import torchvision
import torchvision.transforms as transforms
import os
import csv
import kornia.augmentation as A
import random
import numpy as np

from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import Dataset
from natsort import natsorted

from io import BytesIO

class ToNumpy:
    def __call__(self, x):
        x = np.array(x)
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=2)
        return x


class ProbTransform(torch.nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):  # , **kwargs):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x

def get_transform(opt, train=True, pretensor_transform=False):
    add_nad_transform = False
    
    if opt.dataset == "trojai":
        return transforms.Compose([transforms.CenterCrop(opt.input_height),transforms.ToTensor()])

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
    if (opt.set_arch is not None) and (("nole" in opt.set_arch) or ("mnist_lenet" in opt.set_arch)):
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
    else:
        if opt.dataset == "cifar10":
            transforms_list.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
            if add_nad_transform:
                transforms_list.append(Cutout(1,9))
        elif opt.dataset == "mnist":
            transforms_list.append(transforms.Normalize([0.5], [0.5]))
            if add_nad_transform:
                transforms_list.append(Cutout(1,9))
        elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
            pass
        elif opt.dataset == "imagenet":
            transforms_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
            if add_nad_transform:
                transforms_list.append(Cutout(1,9))
        else:
            raise Exception("Invalid Dataset")
    return transforms.Compose(transforms_list)
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        #print(img)

        return img

class PostTensorTransform(torch.nn.Module):
    def __init__(self, opt):
        super(PostTensorTransform, self).__init__()
        self.random_crop = ProbTransform(
            A.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop), p=0.8
        )
        self.random_rotation = ProbTransform(A.RandomRotation(opt.random_rotation), p=0.5)
        if opt.dataset == "cifar10":
            self.random_horizontal_flip = A.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

def get_dataloader(opt, train=True, pretensor_transform=False, shuffle=True, return_dataset = False):
    transform = get_transform(opt, train, pretensor_transform)
    if opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform=transform, download=True)
    else:
        raise Exception("Invalid dataset")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=shuffle)
    if return_dataset:
        return dataset, dataloader, transform
    else:
        return dataloader, transform

def get_dataloader_random_ratio(opt, train=True, pretensor_transform=False, shuffle=True):
    transform = get_transform(opt, train, pretensor_transform)
    if opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform=transform, download=True)
    else:
        raise Exception("Invalid dataset")
        
    idx = random.sample(range(dataset.__len__()),int(dataset.__len__()*opt.random_ratio))
    dataset = torch.utils.data.Subset(dataset,idx)
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=shuffle)
    return dataloader, transform

def main():
    pass


if __name__ == "__main__":
    main()
