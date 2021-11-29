# import resources
import os, os.path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from torchvision import datasets, models, transforms
import torchvision.transforms as transforms
from Conv_Layer import ConvLayer
from Primary_Caps import PrimaryCaps
from Capsule_Network import CapsuleNetwork
from Capsule_Loss import CapsuleLoss
from Train import train
from Test import test

# check if CUDA is available
TRAIN_ON_GPU = torch.cuda.is_available()

if(TRAIN_ON_GPU):
    print('Training on GPU!')
else:
    print('Only CPU available')

# The data_dir should be replaced with your own local path for hieroglyph image folder downloaded from my github
downloads_path = str(Path.home() / "Downloads")
data_dir = downloads_path + '/EgyptianHieroglyphDataset_Original_Clean/'

train_dir = os.path.join(data_dir, 'train/')
test_dir = os.path.join(data_dir, 'test/')

# classes are folders in each directory with these names
classes = []

for filename in os.listdir(train_dir):
    classes.append(filename)

classes.sort()

data_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Grayscale(num_output_channels=1),
                                          transforms.Resize((75, 75)),
                                          transforms.Normalize((0.5,), (0.5,))
                                          ])

train_data = datasets.ImageFolder(train_dir, transform=data_transform)
test_data = datasets.ImageFolder(test_dir, transform=data_transform)

# define dataloader parameters
batch_size = 20
num_workers=0

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                          num_workers=num_workers, shuffle=True)

# instantiate and print net
capsule_net = CapsuleNetwork()

print(capsule_net)

# move model to GPU, if available
if TRAIN_ON_GPU:
    capsule_net = capsule_net.cuda()

# custom loss
criterion = CapsuleLoss()

# Adam optimizer with default params
optimizer = optim.Adam(capsule_net.parameters())

# Exponential Decay to strengthen learning
decayRate = 0.999
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

# training for 5 epochs
n_epochs = 30
losses = train(capsule_net, criterion, optimizer, n_epochs=n_epochs, train_loader=train_loader)

# call test function and get reconstructed images
caps_output, images, reconstructions = test(capsule_net, test_loader, classes=classes)