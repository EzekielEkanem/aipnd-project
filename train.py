# Import all necessary packages
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from collections import OrderedDict
import json
from get_input_for_train import get_input_args

# Get arguments passed from get_input_args
in_args = get_input_args()

# get directory to train 
data_dir = in_args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#Define the transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(244),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(244),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Define the datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)

test_datasets = datasets.ImageFolder(test_dir, transform=train_transforms)

valid_datasets = datasets.ImageFolder(valid_dir, transform=train_transforms)

# Define the image datasets and the trainforms, define the dataloaders
trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)

testloaders = torch.utils.data.DataLoader(test_datasets, batch_size=32)

validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=32)

def download_model(arch):
    #download model with pre-trained values
    if (arch.lower() == "resnet50"):
            model = models.resnet50(pretrained=True)
    else:
            model = models.densenet121(pretrained=True)
    #Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    return model

def set_model(arch, hidden_units):
    #download model with pre-trained values
    model = download_model(arch)
    #Set model to be used based on user's input
    if (arch.lower() == "resnet50"):
        
        #Define a new feedforward network to be used as a classifier
        classifier = nn.Sequential(OrderedDict([
                                   ("fc1", nn.Linear(2048, hidden_units)),
                                   ("relu1", nn.ReLU()),
                                   ("dropout1", nn.Dropout(p=0.5)),
                                   ("fc2", nn.Linear(hidden_units, 102)),
                                   ("output", nn.LogSoftmax(dim=1))
                                   ]))

        model.fc = classifier
        
    else:
        
        #Define a new feedforward network to be used as a classifier
        classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(1024, hidden_units)),
                            ('relu1', nn.ReLU()),
                            ('dropout1', nn.Dropout(0.2)),
                            ('fc2', nn.Linear(hidden_units, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
        
        model.classifier = classifier
        
    return model
        
model = set_model(in_args.arch, int(in_args.hidden_units)) 
    
#Use GPU if available
if in_args.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu") 

#Set our loss function, optimizer, and move model parameters to GPU or CPU
criterion = nn.NLLLoss()
learning_rate = float(in_args.learning_rate) 
if (in_args.arch.lower() == "resnet50"): 
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
else:
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
model.to(device)

#Train model
epoch = int(in_args.epochs)
print_every = 5

for e in range(epoch):
    running_loss = 0
    for images, labels in trainloaders:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model.forward(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if (e % print_every == 4):
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            model.eval()
            for images, labels in validloaders:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss

                ps = F.softmax(outputs, dim=1)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        model.train()

        print("Epoch: {}/{}.. ".format(e+1, epoch),
              "Training Loss: {:.3f}.. ".format(running_loss),
              "Test Loss: {:.3f}.. ".format(test_loss/len(validloaders)),
              "Test Accuracy: {:.5f}.. ".format(accuracy/len(validloaders)))

        running_loss = 0
        

#Switch to cpu
model.to("cpu")

#Attach mapping of class to indices to the model
model.class_to_idx = train_datasets.class_to_idx

# Save the checkpoint
checkpoint = {"input size": (2048 if (in_args.arch.lower() == "resnet50") else 1024),
              "output size": 102,
              "hidden layers": int(in_args.hidden_units),
              "state_dict": model.state_dict(),
              "class_to_idx": model.class_to_idx,
              "num_of_epochs": epoch,
              "optimizer_state": optimizer.state_dict}

torch.save(checkpoint, in_args.save_dir + "/checkpoint.pth")