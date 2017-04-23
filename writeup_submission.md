#**Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report




[image1]: ./WriteUpPictures/chart1.png "Visualization"
[image2]: ./WriteUpPictures/visualization.png "Visualization 2"
[image3]: ./WriteUpPictures/color.png "Color"
[image4]: ./WriteUpPictures/WebImages/1.png "Traffic Sign 1"
[image5]: ./WriteUpPictures/WebImages/2.png "Traffic Sign 2"
[image6]: ./WriteUpPictures/WebImages/3.jpg "Traffic Sign 3"
[image7]: ./WriteUpPictures/WebImages/4.png "Traffic Sign 4"
[image8]: ./WriteUpPictures/WebImages/5.png "Traffic Sign 5"
[image9]: ./WriteUpPictures/grayscale.png "Grayscale"
[image10]: ./WriteUpPictures/comparison.png "Comparison"
[image11]: ./WriteUpPictures/WebImages/6.jpg "Traffic Sign 5"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/vikasiyer/carnd-term1-p2)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32X3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third and fourth code cell of the IPython notebook.  

Here are exploratory visualization of the data set:

![alt text][image1]
![alt text][image2]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fifth through twelfth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because the color channel does not contribute much in convolutional neural network learning.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]
![alt text][image9]



####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I used the testing, validation and test data provided in the project. I did not augment any additional data. This will be a possible future experiment.



####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the sixth cell of the ipython notebook.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Dropout					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 28x28x6     |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flattening	      	| Output = 400 				|
| Fully connected	1	| Output = 120        									|
| RELU					|												|
| Fully connected	2	| Output = 84        									|
| RELU					|												|
| Dropout					|												|
| Fully connected	3	| Output = 43        									|




####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located from seventh through twelfth cells of the ipython notebook.

To train the model, I used an the following hyperparameters:
Epoch Size = 90
Batch Size = 128
Learning Rate = 0.0008
Optimizer = Adam

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the twelfth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.987
* validation set accuracy of 0.943
* test set accuracy of 0.934

I iterated with several hyper parameters to arrive at the numbers shown above:
* At first, I started with the Lenet-5 architecture as suggested in the project guidelines. The validation accuracy was around the range of 0.87-0.89. I had used the following parameters:
    - EPOCHS:10
    - BATCH SIZE: 128
    - Learning Rate: .001


* To improve the validation accuracy, I tried the following adjustments to architecture and tuned parameters as described below:
  - Added a preprocessing step of changing color image to grayscale.
  - I started tuning the hyper parameters. I started playing with the EPOCH values - and increased it to 50, 100 and then to 90. I plotted a graph to see the inflection point of EPOCH when the graph begins to plateau.  I also tried increasing the BATCH size from 128 to 192 and found that the accuracy improved. However, I could see that the while the training accuracy was moving closer to 100%, the validation accuracies improved only by about 3%.
  - I then tried add 'dropout' layers to reduce overfitting. I also reduced the learning rate and increased the EPOCH size. This step took the accuracy rate above 0.93.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

I based my architecture on the Lenet-5 architecture. Lenet-5 is relevant to the traffic sign application as it is used to classify images. Given that Traffic sign application consists of training a huge dataset of images, convolutional layers help to share parameters.  

I took Lenet-5 and tweaked it further to add a few dropout layers and tuned the hyperparameters to get a more desirable accuracy.

I tried adding a "dropout" layer after every RELU layer (4 of them). This brought down the accuracy significantly. When I reduced the number of dropouts from 4 to 2, the accuracy improved. In my current architecture, I have a dropout after the first RELU layer and after the last RELU layer.

Lowering the learning rate, increasing the EPOCH and introducing a few dropouts in between the other layers seemed to be the key steps.

As shown in the IPython notebook, the following chart shows the "Train vs Validation accuracy" plot. Given that the accuracy values are above 0.93, it demonstrates that the model is working well.

![alt text][image10]



###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I decided to actually pick 6 images. Here are the German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8] ![alt text][image11]

The sixth image might be difficult to classify because it is captured at an angle and the training set may not have a picture of the stop sign captured at an angle. So I picked this to see if the model can handle this.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 22nd and 24th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign				| Speed limit (80km/h)											|
| Bumpy Road      		| Bumpy Road  									|
| Children Crossing     			| 	Speed limit (80km/h)									|
| Speed limit (70km/h)				|   Speed limit (80km/h) 											|
| No vehicles      		| Speed limit (80km/h)					 				|
| 	Stop Sign		|    Roundabout mandatory   							|


The model was able to correctly guess 1 of the 6 traffic signs, which gives an accuracy of 16.66%. This compares unfavorably to the accuracy on the test set. Perhaps the quality of my images were not good enough and not similar enough to those in the training set. Or they were able to match well at the initial layers, but didn't do well in the deeper layers. This looks like a shortcoming in the model.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 22nd and 24th cell of the Ipython notebook.

For the second image, the model is very sure that this is a bumpy road (probability of 1.0), and the image does contain a Bumpy Ride. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00000000e+00         			| Bumpy Road   									|
| 1.21212117e-14					 				|Wild animals crossing
| 1.11301798e-14     				|Road work |
| 1.21584390e-15					| Bicycles crossing											|
| 4.17307325e-17	      			| Turn left ahead					 				|


For all the other images, even the topmost predictions have low probabilities. For eg, for one of the stop sign images, here are the probabilities:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 3.57355982e-01         			| Speed limit (80km/h)   									|
| 2.51989603e-01				 				|Speed limit (60km/h)
| 8.08473751e-02     				|Speed limit (100km/h) |
| 6.13560453e-02					| Speed limit (120km/h)											|
| 3.94520760e-02      			| Speed limit (30km/h)					 				|

Instead of predicting as a stop sign, this is predicting it to be different speed limit signs! So, seems like the model is going by the outward structure of the signs but failing at one of the deeper layers. Or possibly, the quality of the images have not been up to the mark. This is one of the shortcomings of my model. I hope to address this in future.
