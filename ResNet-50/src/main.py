import os, os.path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torchvision import datasets, models, transforms
from hieroglyph_data_preparation import load_data
from model_training import train_model
from model_testing import test_model

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

##################################################Load Hieroglyph Data##################################################
# The data_dir should be replaced with your own local path for hieroglyph image folder downloaded from my github
downloads_path = str(Path.home() / "Downloads")
data_dir = downloads_path + '/EgyptianHieroglyphDataset_Original_Clean/'

# Number of images processed in a single training
batch_size = 20
num_workers = 0

# The load_data function is from hieroglyph_data_preparation python file
train_loader, test_loader, classes = load_data(data_dir)

##################################################ResNet Model##########################################################
# Whether to extract features with the model
feature_extract = False
# Other selections
loss_function = "cross-entropy"
model_selection = "resnet-50"
optim_selection = "Adam"

# False if you want scratch model, True if you want pretrained model
whether_to_pretrain = True

# Load the model
if model_selection == "resnet-50":
    resnet50 = models.resnet50(pretrained=whether_to_pretrain)
    # Number of features in the last layer of resnet
    n_inputs = resnet50.fc.in_features
    # Add last linear layer (n_inputs -> 40 hieroglyph classes)
    last_layer = nn.Sequential(
                    nn.Linear(n_inputs, len(classes)))
    resnet50.fc = last_layer
    if train_on_gpu:
      resnet50.cuda()
    # Specify optimizer (Adam) and learning rate = 0.001
    if optim_selection == "Adam":
        optimizer = optim.Adam(resnet50.parameters(), lr=0.001)

elif model_selection == "inception-v3":
    inception_v3 = models.inception_v3(pretrained=whether_to_pretrain)
    # Number of features in the last layer of resnet
    n_inputs = inception_v3.fc.in_features
    # Add last linear layer (n_inputs -> 40 hieroglyph classes)
    last_layer = nn.Sequential(
                    nn.Linear(n_inputs, len(classes)))
    inception_v3.fc = last_layer
    if train_on_gpu:
      inception_v3.cuda()
    # Specify optimizer (Adam) and learning rate = 0.001
    if optim_selection == "Adam":
        optimizer = optim.Adam(inception_v3.parameters(), lr=0.001)


# Specify loss function (categorical cross-entropy)
if loss_function == "cross-entropy":
    criterion = nn.CrossEntropyLoss()

# Exponential Decay to strengthen learning
decayRate = 0.999
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

##################################################Training##############################################################
# number of epochs to train the model
n_epochs = 50

if model_selection == "resnet-50":
  # The train_model function is from model_training python file
  resnet50, train_losses = train_model(train_loader, optimizer, resnet50, criterion, my_lr_scheduler, n_epochs)
elif model_selection == "inception-v3":
  # The train_model function is from model_training python file
  inception_v3, train_losses = train_model(train_loader, optimizer, inception_v3, criterion, my_lr_scheduler, n_epochs)

##################################################Testing###############################################################
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# The test_model function is from model_testing python file
test_loss, class_correct, class_total, labels, predictions = test_model(classes, resnet50, test_loader, criterion)

# Test accuracy for each hieroglyph
for i in range(len(classes)):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (classes[i], 100 * class_correct[i] / class_total[i], np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

# Total Test accuracy
print("\nAccuracy: {:.3%}".format(accuracy_score(labels, predictions)))
print("\nPrecision: {:.3%}".format(precision_score(labels, predictions, average = 'weighted')))
print("\nRecall: {:.3%}".format(recall_score(labels, predictions, average = 'weighted')))
print("\nF1-score: {:.3%}".format(f1_score(labels, predictions, average = 'weighted')))