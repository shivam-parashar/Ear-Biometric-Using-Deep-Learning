import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


import glob
from numpy import array
#USING PILL
from PIL import Image


filelist = glob.glob('ltrain/*.bmp')

X_train = []
for f1 in filelist:
    gray = cv2.imread(f1)
    img_gaussian = cv2.GaussianBlur(gray,(3,3),0)
    img_sobelx = cv2.Sobel(img_gaussian,cv2.CV_8U,1,0,ksize=5)
    img_sobely = cv2.Sobel(img_gaussian,cv2.CV_8U,0,1,ksize=5)
    img_sobel = img_sobelx + img_sobely
    X_train.append(img_sobel)


X_train = np.array(X_train)

#X_train = np.array([np.array(Image.open(fname)) for fname in filelist])

y_train = []

for i in xrange(0,2):
	y_train.append(i)
	y_train.append(i)
	y_train.append(i)
	

y_train = np.array(y_train)


filelist = glob.glob('ltest/*.bmp')

X_test = []
for f1 in filelist:
    gray = cv2.imread(f1)
    img_gaussian = cv2.GaussianBlur(gray,(3,3),0)
    img_sobelx = cv2.Sobel(img_gaussian,cv2.CV_8U,1,0,ksize=5)
    img_sobely = cv2.Sobel(img_gaussian,cv2.CV_8U,0,1,ksize=5)
    img_sobel = img_sobelx + img_sobely
    X_test.append(img_sobel)


X_test = np.array(X_test)


#X_test = np.array([np.array(Image.open(cv2.Canny(fname,50,100))) for fname in filelist])


y_test = []
for i in xrange(0,2):
	y_test.append(i)
	

y_test = np.array(y_test)



X_train = X_train.reshape(X_train.shape[0], 1, 100, 270).astype('float32') #last 3 arguments are input_shape
X_test = X_test.reshape(X_test.shape[0], 1, 100, 270).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

y_train = np_utils.to_categorical(y_train).astype('int32') #This was commented earlier
y_test = np_utils.to_categorical(y_test).astype('int32')  #This was commented earlier



model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(1, 100, 270), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(15, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='sigmoid'))




# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=7, batch_size=2)


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)


print("Accuracy of CNN is: %.2f%%" % (scores[1]*100))


