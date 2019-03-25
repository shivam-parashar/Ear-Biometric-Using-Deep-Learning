# Ear-Biometric-Using-Deep-Learning
Implementing a 2D Ear Biometrics system that uses Deep Learning for classification.

The problem statement I worked on was to implement a 2D Ear Biometrics system that uses Deep Learning for classification.
The use of Ear images for Biometric purposes is important in the field of biometrics because ear images are more unique than most other features and also are structurally stable at the same time. 
I implemented this using neural networks because neural networks use lesser space and once they are trained, they are also faster than most classification algorithms that do not use neural networks.
I have used SIFT/SURF/ORB from OpenCV for the feature extraction part and keras, tensorflow libraries for the implementation of the neural network. 

Data Collected: 
Data set1: I have captured photos of 50 subjects for this. 12 images per subject.The camera used was Nikon D5300, 24.2MP. 
Data set2: IIT Delhi dataset
Data set3: Spain University Dataset

Results(On data set1 ):
Method VS Accuracy obtained
HOG feature extraction : 77.8 %
Geometrical features using Canny edge detection : 82.5 %
Bag of  features(BOF) + KNN : 79.8 %
Current Method- SURF with bidirectional LSTM : 84.0%
