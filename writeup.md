 # **Traffic Sign Recognition** 

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

 [image1]: ./writeup_images/visualization1.png "Visualization - occurences"
 [image2]: ./writeup_images/visualization2.png "Visualization - sign classes"

 ## Rubric Points
 ### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

 ---
 ### Writeup / README

 #### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

 The submitted workspace contains the required files:
 * CarND-Traffic-Sign-Classifier.ipynb
 * report.html
 * writeup.md

 ### Data Set Summary & Exploration

 #### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

 The data set contains cropped photos taken from german traffic signs converted to small rgb images.
 In the code cell #3 I made a report describing the size input data set:

 | Set           | Size         |
 | ------------- |-------------:|
 | Training      | 34799        |
 | Validation    |  4410        |
 | Test          | 12360        |
 | External test |     7        |

 Properties of each image are 32x32 24bit RGB, so shape is (32,32,3)

 The number of unique sign classes is 43

 #### 2. Include an exploratory visualization of the dataset.

 I have made two visualization image. The first one shows each sign class, the second one describes the data set population by unique classes

 Unique classes
 ![alt text][image2]

 Occurences by unique classes in each set
 ![alt text][image1]

 ### Design and Test a Model Architecture

 #### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

 Pre processing has one step, data was normalized in code cell #5. Normailizing data lowers the impact of rounding errors. Normalized data has a mean of zero, because it increases the efficiency of training with gradient descent.

 #### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

 The model was written in code cells #6-12. I have chosen LeNet-5 architecture described in previous lessons. There are no changes in layer count but in output shapes.

 | Layer         		      |     Description	        					                 | 
 |:---------------------:|:---------------------------------------------:| 
 | Input         		      | 32x32x3 RGB image   							                   |  
 | Convolution 5x5     	 | 1x1 stride, valid padding, outputs 28x28x12 	 |
 | RELU					             |												                                   |
 | Max pooling	      	   | 2x2 stride,  outputs 14x14x12 				            |
 | Convolution 5x5	      | 1x1 stride, valid padding, outputs 10x10x32   |
 | RELU					             |												                                   |
 | Max pooling	      	   | 2x2 stride,  outputs 800 (5x5x32 flatten)     |
 | Fully connected		     | W,b shapes are (800,240),(240) , outputs 240  |
 | Fully connected		     | W,b shapes are (240,84),(84) , outputs 84     |
 | Fully connected		     | W,b shapes are (84,43),(43) , outputs 43      |
 | Softmax classification| Fisrt cross entropy, then softmax was calculated between logits and one hot labels|
 | Adam optimizer | Error minimizing was done using adam_optimizer |

 #### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
 
 I used Adam optimizer for optimizing my models output. It is based on adam algorythm, an extension of Stochastic Gradient Descent. The table below describes the hyperparameters:
 | Parameter name | Description	| Value | Tuning considerations |
 |:--------------:|:-----------:|:-----:|:-------:|
 |epochs|The number of training runs on each batch| 36|
 |batch size|The number of images in each batch| 512 |
 |mu|The mean of truncate mean distribtion used in W matrices| 0 |
 |sigma|The standard deviation of the normal distribution| 0.1 | 
 |rate|0.0028|learing rate of optimizing algorytm| 


 #### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I have started with LeNet-5 and kept going with it not changing the architecture but the shapes of layers. I have chosen it because it uses convolution layers suitable for classifying statistical invariant objects like traffic signs.

My final model results were:
 * Validation set accuracy of 0.957
 * Test set accuracy of 0.945
 * External test set accuracy of 0.8

I found that the depth of first and second convolution layer is not enough for a 3-depth input, so I modified them to 16 and 32, rescpectively. Fully connected layer inputs were adepted to them.

Validation accuracy on consecutive epochs was alternating around a slightly emerging result, so I set a large number on it
 Increasing the batch size from 128 to 512 resulted about 1% bigger validation accuracy
 
Mean of distribution being in the center helps gradient descent to work better. Decreasing and increasing the width of distribution caused lower accuracy, so I left in unchanged
 
Learning rate was set to 0.01 first. With this value error decreased very fast but the increasing of validation accuracy suddenty stopped after a few epochs, so I lowered learning rate and set the number of epochs bigger. These two settings caused approx 3% boost in validation accuracy |
 

 If an iterative approach was chosen:
 * What was the first architecture that was tried and why was it chosen?
 * What were some problems with the initial architecture?
 * How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
 * Which parameters were tuned? How were they adjusted and why?
 * What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

 If a well known architecture was chosen:
 * What architecture was chosen?
 * Why did you believe it would be relevant to the traffic sign application?
 * How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


 ### Test a Model on New Images

 #### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

 Here are five German traffic signs that I found on the web:

 ![alt text][image4] ![alt text][image5] ![alt text][image6] 
 ![alt text][image7] ![alt text][image8]

 The first image might be difficult to classify because ...

 #### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

 Here are the results of the prediction:

 | Image			        |     Prediction	        					| 
 |:---------------------:|:---------------------------------------------:| 
 | Stop Sign      		| Stop sign   									| 
 | U-turn     			| U-turn 										|
 | Yield					| Yield											|
 | 100 km/h	      		| Bumpy Road					 				|
 | Slippery Road			| Slippery Road      							|


 The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

 #### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

 The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

 For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

 | Probability         	|     Prediction	        					| 
 |:---------------------:|:---------------------------------------------:| 
 | .60         			| Stop sign   									| 
 | .20     				| U-turn 										|
 | .05					| Yield											|
 | .04	      			| Bumpy Road					 				|
 | .01				    | Slippery Road      							|


 For the second image ... 

 ### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
 #### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


