import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation,LSTM, Bidirectional, TimeDistributed
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from scipy import stats
from keras import backend as K
import matplotlib.pyplot as plt
K.set_image_dim_ordering('th')



def normalize(X):
	for j in xrange(0,X.shape[1]):
		for k in xrange(0,X.shape[2]):
			mean = np.mean(X[:,j,k],axis=None)
			stddev = np.std(X[:,j,k],axis=None)
			if stddev == 0:
				X[:,j,k] = mean
			else:
				X[:,j,k] = (X[:,j,k]-mean)/stddev
	return X

# Initiate SIFT detector
#sift = cv2.SIFT()
sift = cv2.xfeatures2d.SIFT_create()

k_max =300

filelist = glob.glob('atrain/*.JPG')

X_train = []
y_train = []
for f1 in filelist:
	print(f1)
	img = cv2.imread(f1,0)
	Kp, des = sift.detectAndCompute(img,None)
	Des = np.zeros((k_max,des.shape[1]))
	Des[:des.shape[0],:des.shape[1]] = des
	X_train.append(Des)
	y_train.append(int(f1[7:].split('_')[0]))
 


X_train = np.array(X_train).astype('float32')
y_train = np_utils.to_categorical(y_train).astype('int32')

filelist = glob.glob('atest/*.JPG')

X_test = []
y_test = []

for f1 in filelist:
	img = cv2.imread(f1,0)
	print(f1)
	Kp, des = sift.detectAndCompute(img,None)
	Des = np.zeros((k_max,des.shape[1]))
	Des[:des.shape[0],:des.shape[1]] = des
	X_test.append(Des)
	y_test.append(int(f1[6:].split('_')[0]))

X_test = np.array(X_test).astype('float32')
y_test = np_utils.to_categorical(y_test).astype('int32')

X_train = normalize(X_train)
X_test = normalize(X_test)

X_train = X_train.reshape(X_train.shape[0],k_max,128)
X_test = X_test.reshape(X_test.shape[0],k_max,128)

print(y_test)

print(X_train[0].shape)

def lstm_model(time_steps,hidden_size,no_tags):
    model=Sequential()
    model.add(LSTM(units=hidden_size,input_shape=(time_steps,hidden_size),return_sequences=True, kernel_initializer="glorot_normal", recurrent_initializer="glorot_normal", activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(units=hidden_size,input_shape=(time_steps,hidden_size),return_sequences=True, kernel_initializer="glorot_normal", recurrent_initializer="glorot_normal", activation='sigmoid')))
    model.add(Dropout(0.2))
    #model.add(Bidirectional(LSTM(units=hidden_size,input_shape=(time_steps,hidden_size),return_sequences=True, kernel_initializer="glorot_normal", recurrent_initializer="glorot_normal", activation='sigmoid')))
    #model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(units=hidden_size,input_shape=(time_steps,hidden_size),return_sequences=False, kernel_initializer="glorot_normal", recurrent_initializer="glorot_normal", activation='sigmoid')))
    model.add(Dropout(0.2))
    model.add(Dense(no_tags))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = lstm_model(k_max,128,y_train.shape[1])
print(model.summary())


history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=1)

#Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy of LSTM is: %.2f%%" % (scores[1]*100))
print(y_test)

precision = history.history['val_precision'][0]
recall = history.history['val_recall'][0]
f_score = (2.0 * precision * recall) / (precision + recall)
print 'F1-SCORE {}'.format(f_score)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()





