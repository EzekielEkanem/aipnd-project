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
#          provide some or all of the 5 inputs, then the default values are
#          used for the missing inputs. Command Line Arguments:
#     1. Image data as image_path with default value 'flowers/test/1/image_06764.jpg'
#     2. Checkpoint as checkpoint with default value '.'
#     3. Top K values as --topk with default value '5'
#     4. Category Names as --category_names with default value 'cat_to_name.json'
#     5. GPU as --gpu with default value 'False'

def get_input_pred():
    
    """
    Create a function that retrieves the following command line inputs 
         from the user using the Argparse Python module. If the user fails to 
         provide some or all of the 5 inputs, then the default values are
         used for the missing inputs. Command Line Arguments:
    1. Image data as image_path with default value 'flowers/test/1/image_06764.jpg'
    2. Checkpoint as checkpoint with default value '.'
    3. Top K values as --topk with default value '5'
    4. Category Names as --category_names with default value 'cat_to_name.json'
    5. GPU as --gpu with default value 'False'
    """
    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create 5 command line arguments as mentioned above using add_argument() from ArgumentParser method
    parser.add_argument('image_path', type = str, default = 'flowers/test/1/image_06764.jpg', help = 'Pass in path to image')
    parser.add_argument('checkpoint', type = str, default = '.', help = 'checkpoint directory')
    parser.add_argument('--topk', type = str, default = '5', help = 'Return top K most likely classes')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'Use a mapping of categories to real names')
    parser.add_argument('--gpu', type = str, default = 'False', help = 'Use GPU for inference')
    
    return parser.parse_args()