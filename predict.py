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
from get_input_for_predict import get_input_pred

# Function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if (checkpoint["input size"] == 2048):
        model = models.resnet50(pretrained=True)
    else:
        model = models.densenet121(pretrained=True)
    classifier = nn.Sequential(OrderedDict([
                           ("fc1", nn.Linear(checkpoint["input size"], checkpoint["hidden layers"])),
                           ("relu1", nn.ReLU()),
                           ("dropout1", nn.Dropout(p=0.5)),
                           ("fc2", nn.Linear(checkpoint["hidden layers"], checkpoint["output size"])),
                           ("output", nn.LogSoftmax(dim=1))
                           ]))

    if (checkpoint["input size"] == 2048):
        model.fc = classifier
    else:
        model.classifier = classifier
    
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]
    model.num_of_epochs = checkpoint["num_of_epochs"]
    model.optimizer_state = checkpoint["optimizer_state"]
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Process a PIL image for use in a PyTorch model
    with Image.open(image) as im:
        (width, height) = (im.width, im.height)
        if (width > height):
            resized_im = im.resize((round(width/height*256), round(256)))
        else:
            resized_im = im.resize((round(256), round(height/width*256)))

        (width, height) = (resized_im.width, resized_im.height)
        left = (width - 224) / 2
        upper = (height - 224) / 2
        right = (width + 224) / 2
        lower = (height + 224) / 2
        cropped_im = resized_im.crop((left, upper, right, lower))

        np_image = np.array(cropped_im) / 255

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (np_image - mean) / std

        image = image.transpose((2, 0, 1))

        return image
    
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Make code run on GPU
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    # Implement the code to predict the class from an image file
    #Process the image
    processed_image = process_image(image_path)
    processed_image = torch.from_numpy(processed_image)
    processed_image = processed_image.unsqueeze(0)
    processed_image = processed_image.type(torch.FloatTensor)
    processed_image = processed_image.to(device)

    #Process the model checkpoint
    model = load_checkpoint(model)
    
    model.to(device)
    model.eval()

    #Predict the class
    output = model(processed_image)
    ps = F.softmax(output, dim=1)
    top_p, top_class = ps.topk(5, dim=1)

    #Invert the class to index dictionary
    model.idx_to_class = {}
    for cl, idx in model.class_to_idx.items():
        model.idx_to_class["{}".format(idx)] = cl

    top_class = [model.idx_to_class["{}".format(c)] for c in top_class[0]]
    # top_prob = top_p[0].detach().numpy()
    # top_prob = [prob for prob in top_prob]

    return (top_p, top_class)

# Get arguments passed from get_input_pred
in_args = get_input_pred()

category_names = in_args.category_names

#Map category label to category name
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

# Get the checkpoint directory
checkpoint = in_args.checkpoint
checkpoint = checkpoint + "/checkpoint.pth"
    
#Call the predict function to predict the image
probs, classes = predict(in_args.image_path, checkpoint, in_args.topk, in_args.gpu)

top_prob = probs.cpu().detach().numpy()
classes = [cat_to_name[c] for c in classes]

print("Class name (in probabilities): \n")

for i in range(len(top_prob[0])):
    print(classes[i] + "({})".format(round(top_prob[0][i], 3)))