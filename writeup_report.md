# **Behavioral Cloning** 

## Writeup by Marcus Neuert (2018-08-12)

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/loss_visualization.png "Model Visualization"
[image2]: ./examples/training_data_distribution.png "Training data distribution"

[image3]: ./examples/input_image_BGR.png "Original Center Image imported by opencv in BGR"
[image4]: ./examples/input_image_RGB.png "Original Center Image converted to RGB"

[image13]: ./examples/preprocessed1.png "Flipped Center Image"
[image14]: ./examples/preprocessed2.png "Negative Gamma Center Image"
[image15]: ./examples/preprocessed3.png "Positive Gamma Center Image"
[image5]: ./examples/preprocessed4.png "Brigth Center Image"
[image6]: ./examples/preprocessed5.png "Dark Center Image"

[image7]: ./examples/preprocessed6.png "Original Left Image"
[image20]: ./examples/preprocessed7.png "Flipped Left Image"
[image16]: ./examples/preprocessed8.png "Negative Gamma Left Image"
[image17]: ./examples/preprocessed9.png "Positive Gamma Left Image"
[image8]: ./examples/preprocessed10.png "Bright Left Image"
[image9]: ./examples/preprocessed11.png "Dark Left Image"

[image10]: ./examples/preprocessed12.png "Original Right Image"
[image21]: ./examples/preprocessed13.png "Flipped Right Image"
[image18]: ./examples/preprocessed14.png "Negative Gamma Rigth Image"
[image19]: ./examples/preprocessed15.png "Positive Gamma Rigth Image"
[image11]: ./examples/preprocessed16.png "Bright Right Image"
[image12]: ./examples/preprocessed17.png "Dark Right Image"

[image22]: ./examples/center_2018_07_28_22_22_09_994.jpg "Recovery from right Image"
[image23]: ./examples/center_2018_07_28_22_22_40_995.jpg "Recovery from left Image"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The code contains a lot of switches and parameters to configure the execution, as you see below. I also implemented generators to train the network, but I trained the network without them, as I had a better result without.

```sh
# network parameters
TURNING_OFFSET = 0.25
LIMIT_IMAGES_PER_TURNING_ANGLE = 400
TRAIN_VALID_SPLIT = 0.2
TRAIN_EPOCHS = 5
LEARN_RATE = 0.0001

# train with generators
BATCH_SIZE = 512
USE_GENERATOR = False

# training image editing
FLIP_IMAGES = False
ADAPT_BRIGHTNESS = True
USE_GAMMA_CORRECTION = False
USE_TRACK2 = True

# debug settings
DEBUG = True
LIMIT_IMAGES_FOR_DEBUGGING = 40000
```

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network (derived from LeNet) with three 5x5 and two 3x3 filter sizes and depths between 24 and 64 (model.py lines 204-243) 

The model includes RELU layers to introduce nonlinearity , and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer with the possibility to tune the learning rate. By lowering the learning rate (from 0.001 to 0.0001) I achieved, that the vehicle made slower turns and didn't drive over lanes or edges smoothly.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and the second track.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because it brought a very good result after the first run without major changes. So I was sure, it will work with only some small modifications.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I added 3 Dropout-Layers. One after the 3 layers with 5x5 filters, one after the two layers with 3x3 filters and one after the first Dense layer.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. Mostly because of shadow or sunlight on the road. To improve the driving behavior in these cases, I used brightness adjustment. I also played around with flipped images and gamma correction. But the best combination was using brightness adjustments and images from track 2.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. Also track 2 is almost perfectly driven.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

```sh
Keras layer summary (created by model.summary() ): 
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
cropping2d_2 (Cropping2D)    (None, 80, 320, 3)        0         
_________________________________________________________________
lambda_2 (Lambda)            (None, 80, 320, 3)        0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 38, 158, 24)       1824      
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 17, 77, 36)        21636     
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 7, 37, 48)         43248     
_________________________________________________________________
dropout_4 (Dropout)          (None, 7, 37, 48)         0         
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 5, 35, 64)         27712     
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 3, 33, 64)         36928     
_________________________________________________________________
dropout_5 (Dropout)          (None, 3, 33, 64)         0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 6336)              0         
_________________________________________________________________
dense_5 (Dense)              (None, 1000)              6337000   
_________________________________________________________________
activation_4 (Activation)    (None, 1000)              0         
_________________________________________________________________
dropout_6 (Dropout)          (None, 1000)              0         
_________________________________________________________________
dense_6 (Dense)              (None, 100)               100100    
_________________________________________________________________
activation_5 (Activation)    (None, 100)               0         
_________________________________________________________________
dense_7 (Dense)              (None, 10)                1010      
_________________________________________________________________
activation_6 (Activation)    (None, 10)                0         
_________________________________________________________________
dense_8 (Dense)              (None, 1)                 11        
=================================================================
Total params: 6,569,469
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

First image imported with opencv standard settings as BGR, second image and following images after converting to RGB.

![alt text][image3] 
![alt text][image4]

Afterwards, I used the left and right camera images, which look like these:

![alt text][image7]
![alt text][image10]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive back to the center of the road as soon as possible. These images show what a recovery looks like starting from :

![alt text][image23]
![alt text][image22]

Then I repeated this process on track two in order to get more data points.

After the collection process, I had more than 10,000 of data points. I then preprocessed this data by the following methods.

## Image Augmentation

To augment the data set, I also changed the brightness of the images. For all images (center, left and right) I created one bright and one dark image:

![alt text][image5]
![alt text][image6]

![alt text][image8]
![alt text][image9]

![alt text][image11]
![alt text][image12]

Because track 2 has enough left and right turns, I figured out, the model trains well without flipping the images. Nevertheless, the implementation for flipping images and gamma correction is working, but optional.

Flipped Center:

![alt text][image4]
![alt text][image13]

Flipped Left:

![alt text][image7]
![alt text][image20]

Flipped Right:

![alt text][image10]
![alt text][image21]


Gamma Correction Center:

![alt text][image14]
![alt text][image15]

Gamma Correction Left:

![alt text][image16]
![alt text][image17]

Gamma Correction Right:

![alt text][image18]
![alt text][image19]

I limited the images per turning angle to 400, which is still very high, in order to have a little balance between the images. Because I didn't use the generator apporach, I was only able to use 40k images for training and validation, so I had also had to limit the overall images. Therefore I shuffled the csv file already, so that I didn't use the first images from track 1 only. I ended up with this data distribution:

![alt text][image2]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as training and validation loss was equally low. I used an adam optimizer with a learning rate of 0.0001.

Here is a visualization error loss function of training and validation set:

![alt text][image1]

## Impediments during development

- Use of the Activation function together with the Dense-Layers didn't work: model.add(Dense(10, activation="relu")) --> Took me several days
- Use of generators was not faster and not more efficient than without
- Use of flipping images and gamma correction filled up my training data set with unnecessary images of the same turning angle
- Matplotlib made problems when importing too early, because of image output to stdout, when calling from bash
- Different Tensorflow versions and other libraries in the AWS image and the current git repository took me several days to make it work

## Possible improvements

- Use generators with memory and performance optimization
- Crop images directly after import to save memory
- Make vehicle turn slower on track 2 and do not allow the tire on the lane line