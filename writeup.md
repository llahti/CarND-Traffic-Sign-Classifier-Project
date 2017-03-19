# Traffic Sign Recognition

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./illustrations/classid_histogram.png "Class ID Histogram."
[image2]: ./illustrations/sample_images.png "Sample of images in training set."
[image3]: ./illustrations/sample_images_processed.png "Sample of preprocessed images."
[image4]: ./illustrations/data_rotation.png "Data augmentation by rotating image."
[image5]: ./illustrations/data_augmentation_shift.png "Data augmentation by shifting image."
[image6]: ./illustrations/model.png "NeuralNet model"
[image7]: ./illustrations/5_samples_from_web.png "5 German traffic signs from web"
[image8]: ./illustrations/5_samples_from_web_preprocessed.png "Preprocessed german traffic signs"
[image9]: ./illustrations/softmax_propabilities.png "Softmax propabilites for 5 german traffic signs"

[image10]: ./illustrations/vis_L1_conv.png "Layer 1 convolution"
[image11]: ./illustrations/vis_L1_activation.png "Layer 1 activation"
[image12]: ./illustrations/vis_L1_pooling.png "Layer 1 pooling"
[image13]: ./illustrations/vis_L2_conv.png "Layer 2 convolution"
[image14]: ./illustrations/vis_L2_activation.png "Layer 2 activation"
[image15]: ./illustrations/vis_L4_conv.png "Layer 4 convolution"
[image16]: ./illustrations/vis_L4_activation.png "Layer 4 activation"

[image17]: ./illustrations/training_e120_lr0_0005_d0_85.png "Final model: Accuracies and hyperparameters"


# Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

## 1. Submissing files

Submission contains all the files required by rubic.
1. writeup.md (This file)
2. Traffic_Sign_Classifier.ipyn
3. Report.html (generated from Traffic_Sign_Classifier.ipyn)
4. Visualizations in ./illustrations folder
5. Test images in ./test_images folder/
6. sign_names.csv
7. LareNet-6 saved model

## 2. Dataset Exploration


### 1. Loading data

Code for this step is in first code cell of the IPython notebook.

Data was stored in a python pickle file and it contained training, validation and test set.


### 2. The basic summary of the data set.

Code for this step is in second code cell of the IPython notebook.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43



### 3. Exploratory visualization of the dataset

Code for this step is in third and fourth code cells of the IPython notebook.

Here is the sample of images in training set

![alt text][image2]


Here is an exploratory visualization of the data set. It is a histogram showing quantity of traffic sgns in each category.

![alt text][image1]


## 3. Design and Test a Model Architecture

### 1. Preprocessing

In this section i describe how and why I preprocessed images. What tecniques were chosen and why those were chosen. 

Data should be optimized before it is fed to the neural network in order to:

* Make neural network easier to train
  - For example if mean of data is far away from zero then learning algorithm need to do alot of work to fit data
* Better performance
  - For example by using simple image processing we can make features more visible before feeding image to neural nel.
* Generally make date more suitable for neural net

#### Preprocessing pipeline

Code for preprocessing pipeline is located in fifth code cell of the IPython notebook and preprocessing is run in the sixth code cell.

In my code there are following steps in pre-processing pipeline

1. Convert RGB to **grayscale**
   - From grayscale image we are able to extract most of the features
   - Reduce amount of data needed by 3 times
   - It is easier to work with grayscale images
2. Adjust mean to zero
   - Zero mean of data helps neural net to train faster and better
3. Limit range to -1...1
   - By adjusting range of values around 0 it is possible to have zero mean
   - But also it prevents data values going too large which would cause problems when neural net is trained

Preprocessed images are shown below. Traffic signs are same than image above in data Exploration step.
![alt text][image3]


### 2. Model Architecture

Code for model architecture is located in eight code cell of the IPython notebook.

Model consist of single-stage deep neural network where are 6 convolutional layers and 3 fully connected layers. This model is based on the model in LeNet exercice and i have made model more deep and wide for better accuracy. Model structure is explained in below image.

I decide to use ELUs as an activation function as those are working better on deep neural nets (https://arxiv.org/abs/1511.07289)

My final model is constructed as shown in image and it can be found from the IPython notebook's code cell.

![alt text][image6]



### 3. Model Training

Code for model training is located eleventh code cell of the IPython notebook.
Accuracy evaluation function is located in tenth code cell of the IPython notebook.

I have been experimenting model training with and without augmented dataset and varying hyperparameters such as number of epochs, learning rate and batch size.

I had to limit batch size 1024 due to my new GPU was crashing if batch size was bigger. I believe that was due to problems related to GPU utilization rate or power supply.

I used an Adam-Optimizer which was used also in LeNet exercise. Though there were few papers which recommended Stochastic Gradient Descent (SGD) - optimizer over Adam-optimizer, because SGD was able to achieve higher accuracy. Generally Adam-optimizer is faster than SGD so i selected it because in my case time is money and i needed results fast.

I used decaying learning rate setting to make model to converge early to reasonaly accuracy and then for longer training periods allow it slowly gain bit more accuracy

The code for training the model is located in the eigth cell of the ipython notebook. 

 
### 4. Solution Approach

#### 1. Setting up training, validation and test data

Data is splitted into training, validation and test sets. This happens in first code cell when data is loaded from pickle file. It could be also reasonable to combine all datasets and then randomly split those to new training and validation and test sets. Potential reason for that is that i don't have information that have that data already been randomly splitted or not.

It would be also interesting to try out K-fold crossvalidation as it could be done within one day with my current workstation. Need to remember that number of K can't be very high. perhaps K=5 could be still doable within reasonable time to test model's performance sometimes.

Final number of images in sets after data augmentation is:
- Training set 1565955
- Validation set 4410
- Testing set 12630


#### 2. Data augmentation

Code for data augmentatation is located in seventh code cell of the IPython notebook.

I decide to augment data by rotating and shifting. Other techiques such as tilting, stretching, adjusting brightness and adjusting contrast i left out from my project. Data augmentation is good way to get more training data cheaply also it helps to train model better, makes it more robust and prevent overfitting.

Example of image rotation
![alt text][image4]

Example of image shifting
![alt text][image5]


#### 3. Training

Final model was trained by using: 
- augmented training set
- number of epochs were 120
- batch size 512


#### 4. Model evaluation

Code for final model evaluation which calculates accuracy for training, validation and test data sets is located in twelth code cell of the IPython notebook.

The function for calculating accuracy is the original one from LeNet-exercice and it is used in 2 key places. One place to use it is in training loop to evaluate training and validation accuracy. Note that i don't evaluate test date set in training loop because test set should be used sparingly.

The other place is the final evaluation of model where training, validation and test sets are all evaluated

My final model results were:
* Training Accuracy   = 0.999
* Validation Accuracy = 0.994
* Test Accuracy       = 0.980

Following image shows plot of training and validation accuracies during training + hyperparamteres which were used
![alt text][image17]

Bit history of my model...

I started model development from the model introduced in LeNet exercise. In beginning i didn't do any preprocessing for data so images were fed to model as is. With this approach i had problems that sometimes it took many epochs that model started converging to high accuracy. Also the final accuracy was not enough.

Then next step was to make model deeper and wider. I added more convolutional layers for better feature extraction. I did several experiments and finally i was able to achieve about 98% validation accuracy. Accuracy on test set were somewhere around 96%.

Then i implemented dropout on fully connected layers to preven over fitting.

After this i begin to experiment with data preprocessing where i implemented image grayscaling and normalization where image mean is adjusted to zero and range is limited to -1...1. This image preprocessing was one of the key steps to make model training faster. 

If i would change something i would do data preprocessing as a first step as it is one of the most important things to make model successful. Then bit more thought need to be put to neural net structure. Perhaps multi-stage structure would be reasonable. One other possibility is to use already made models such as Inception-V3 or similar.


## 4. Test a Model on New Images

### 1. Choosing images

In my Ipython notebook I load 5 test images from disk in fourteenth code cell,

I tested my model with 5 german traffic signs downloaded from web. Here is picture of those

![alt text][image7]

And same traffic signs after preprocessing

![alt text][image7]

Few comments about these german traffic signs Comments are per image index

1. For human it seems to be obvious that it is 80km/h speed limit sign, but when you look closely the upper part of the eight is bit blurred which may cause problems to detection algorithm.
2. Looks clear, shouldn't be problem
3. Some reflections on sign, may pose a problem
4. Clear sign, shouldn't be problem
5. Clear sign, shouldn't be problem


### 2. Results

Prediction code is run in fifteenth code cell of the IPython notebook.

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Index | Truth | Prediction | Description (Predicted)  |
|:-----:|:-----:|:----------:|:------------------------:|
|    0  |    5  |         5  | Speed limit (80km/h)     |
|    1  |    1  |         1  | Speed limit (30km/h)     |
|    2  |   28  |        28  | Children crossing        |
|    3  |   33  |        33  | Turn right ahead         |
|    4  |   17  |        17  | No entry                 |


First traffic sign on this test set was incorrecly classified depending of training hyperparameters (mostly about epochs). For example when model trained 50 epocs the 80km/h sign was mis-classified as 60km/h. All other traffic signs were predicted correctly. It's bit difficult to say why this happen. If you look number "8" very closely you could see that upper half of the eight could also prepresent number 6 as part of the number 8 could be seen as "noise". This results were obtained by training model 100 epochs.

### 3. Performance on New Images

Performance is analyzed in sixteenth code cell of the IPython notebook.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

Model predicted 5 signs correctly out of 5. Prediction accuracy is 100.0% for 5 test images downloaded from web.


### 4. Model Certainty

Certainty (Topk-5) analysis is in 18th code cell and results are plotted in 19th code cell.
Functions needed for plotting are defined in 17th code cell.

Model is very confident of it's predictions. Only the first image (80km/h limit) has visible predictions on other classes. There it is predicted in 94% probability that it is indeed **80km/h sign** second best propability is ~6% for **60km/h limit**.

![alt text][image9]


### Visualize the Neural Network's State with Test Images

Visualization code is located in 20th and 21st code cells of the IPython notebook

I visualized some convolutions, activations and pooling tensors and below are some of the examples i was able to visualize.

![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]

