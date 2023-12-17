# Flower Classification

This project uses convolutional neural network to train an image classifier that is able to identify 102 different flower species using two deep learning models (resnet50 and densenet121). Both models gave an accuracy of over 70%. This image classifier can be used to identify flower species from new images, e.g., in a phone app that tells you the name of the flower your camera is looking at.

# Problem to Solve
Build an application that tells the name of a flower from an image.

# Available data
102 Category Flower Dataset was given by the Nanodegree program. This dataset contains images of 102 different flower species with lables. These images have different sizes.

Data file structure:

flowers: folder of image data.
* train, valid, test: subfolders for training, validating, and testing the image classifier, respectively.
* 1, 2, ..., 102: 102 subfolders whose names indicate different flower categories. Given the large data size, data folders are not included here.

# What I did
1. Data loading and data preprocessing
   
- [x] **Loaded image data**
- [x] **Training set: Applied transformations such as rotation, scaling, and horizontal flipping (model generalizes / performs better)**
- [x] **All datasets: Resized and cropped to the appropriate image size (required by pre-trained model)**
- [x] **All datasets: Normalized image colors (RGB) using mean and standard deviation of pre-trained model**
- [x] **Training set: Shuffled Data at each epoch**

2. Build and train the model

- [x] **Loaded a pre-trained network resnet50 and freezed parameters**
- [x] **Defined a new, untrained neural network as a classifier. The classifier has a hidden layer (ReLU activation) and an output layer (LogSoftmax activation). Assign dropout to reduce overfitting.**
- [x] **Assigned criterion (NLLLoss, negative log loss) and optimizer (Adam, adaptive moment estimation)**
- [x] **Trained the classifier layers using forward and backpropagation on GPU**
- [x] **Tracked the loss and accuracy on the validation set to determine the best hyperparameters**

3. Use the trained classifier to predict image content

- [x] **Tested the trained model on testing set (71% accuracy)**
- [x] **Saved the trained model as checkpoint**
- [x] **Wrote a function that gives top-5 most probable flower names based on image path**

4. Build a command line application

* See below for details
![](https://github.com/EzekielEkanem/aipnd-project/blob/master/assets/inference_example.png)

# How to run the command line application

**Train the image classifier**

![train.py](https://github.com/EzekielEkanem/aipnd-project/blob/master/train.py): Train the image classifier, report validation accuracy along training, and save the trained model as a checkpoint.

* Basic usage:

  - [x] Specify directory of image data: **python train.py flowers**
* Options:

  - [x] Set directory to save checkpoints: **python train.py flowers --save_dir assets**

  - [x] Choose architecture: **python train.py flowers --arch "vgg13"**

  - [x] Set hyperparameters: **python train.py flowers --learning_rate 0.001 --hidden_units 512 --epochs 20**

  - [x] Use GPU for training: **python train.py flowers --gpu**
 
  **Identify flower name from a new image**

![predict.py](https://github.com/EzekielEkanem/aipnd-project/blob/master/predict.py): Use the trained image classifier to predict flower name along with the probability of that name

* Basic usage:

  - [x] Specify file path of the image and directory name of saved checkpoint: **python predict.py flowers/test/100/image_07896.jpg assets**
* Options:

  - [x] Return top K most likely classes: **python predict.py flowers/test/100/image_07896.jpg assets --top_k 3**
  - [x] Use a mapping of categories to real names: **python predict.py flowers/test/100/image_07896.jpg assets --category_names cat_to_name.json**
  - [x] Use GPU for inference: **python predict.py flowers/test/100/image_07896.jpg assets --gpu**
