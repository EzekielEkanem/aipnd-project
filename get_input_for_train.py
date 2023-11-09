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
import argparse

# Create a function that retrieves the following command line inputs 
#          from the user using the Argparse Python module. If the user fails to 
#          provide some or all of the 7 inputs, then the default values are
#          used for the missing inputs. Command Line Arguments:
#     1. Image data as data_directory with default value 'flowers'
#     2. Checkpoint as --save_dir with default value '.'
#     3. CNN Model Architecture as --arch with default value 'resnet50'
#     4. Learning Rate as --learning_rate with default value '0.003'
#     5. Hidden units as --hidden_units with default value '1000'
#     6. Number of training round as --epochs with default value '25'
#     7. GPU as --gpu with default value 'False'

def get_input_args():
    
    """
    Create a function that retrieves the following command line inputs 
         from the user using the Argparse Python module. If the user fails to 
         provide some or all of the 7 inputs, then the default values are
         used for the missing inputs. Command Line Arguments:
    1. Image data as data_directory with default value 'flowers'
    2. Checkpoint as --save_dir with default value '.'
    3. CNN Model Architecture as --arch with default value 'resnet50'
    4. Learning Rate as --learning_rate with default value '0.003'
    5. Hidden units as --hidden_units with default value '1000'
    6. Number of training round as --epochs with default value '25'
    7. GPU as --gpu with default value 'False'
    """
    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create 7 command line arguments as mentioned above using add_argument() from ArgumentParser method
    parser.add_argument('data_directory', type = str, default = 'flowers', help = 'Set directory to load training data')
    parser.add_argument('--save_dir', type = str, default = '.', help = 'Set directory to save checkpoint')
    parser.add_argument('--arch', type = str, default = 'densenet121', help = 'Check architecture')
    parser.add_argument('--learning_rate', type = str, default = '0.003', help = 'Set learning rate')
    parser.add_argument('--hidden_units', type = str, default = '1000', help = 'Set hidden units')
    parser.add_argument('--epochs', type = str, default = '15', help = 'Set number of epochs')
    parser.add_argument('--gpu', type = str, default = 'False', help = 'Use GPU for training')
    
    return parser.parse_args()