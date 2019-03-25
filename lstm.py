import numpy as np
import cv2
from matplotlib import pyplot as plt
from time import time
import glob
import tensorflow as tf
from keras.callbacks import TensorBoard
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
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as pltra
import keras_metrics

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

k_max =150

filelist = glob.glob('etrain/*.jpg')

X_train = []
y_train = []
for f1 in filelist:
	print(f1)
	img = cv2.imread(f1,0)
	alg = cv2.KAZE_create()
	kps = alg.detect(img)
	kps = sorted(kps, key=lambda x: -x.response)[:150]
	kps, des = alg.compute(img, kps)
	Des = np.zeros((k_max,des.shape[1]))
	Des[:des.shape[0],:des.shape[1]] = des
	X_train.append(Des)
	y_train.append(int(f1[7:].split('_')[0]))
 


X_train = np.array(X_train).astype('float32')
y_train = np_utils.to_categorical(y_train).astype('int32')

filelist = glob.glob('etest/*.jpg')

X_test = []
y_test = []

for f1 in filelist:
	img = cv2.imread(f1,0)
	print(f1)
	alg = cv2.KAZE_create()
	kps = alg.detect(img)
	kps = sorted(kps, key=lambda x: -x.response)[:150]
	kps, des = alg.compute(img, kps)
	Des = np.zeros((k_max,des.shape[1]))
	Des[:des.shape[0],:des.shape[1]] = des
	X_test.append(Des)
	y_test.append(int(f1[6:].split('_')[0]))

X_test = np.array(X_test).astype('float32')
y_test = np_utils.to_categorical(y_test).astype('int32')

X_train = normalize(X_train)
X_test = normalize(X_test)

X_train = X_train.reshape(X_train.shape[0],k_max,64)
X_test = X_test.reshape(X_test.shape[0],k_max,64)

print(y_test)

print(X_train[0].shape)

def lstm_model(time_steps,hidden_size,no_tags):
    model=Sequential()
    model.add(LSTM(units=hidden_size,input_shape=(time_steps,hidden_size),return_sequences=True, kernel_initializer="glorot_normal", recurrent_initializer="glorot_normal", activation='sigmoid'))
    model.add(Dropout(0.2))
	#model.add(Bidirectional(LSTM(units=hidden_size,input_shape=(time_steps,hidden_size),return_sequences=True, kernel_initializer="glorot_normal", recurrent_initializer="glorot_normal", activation='sigmoid')))
	#model.add(Dropout(0.2))
    
    #model.add(Bidirectional(LSTM(units=hidden_size,input_shape=(time_steps,hidden_size),return_sequences=True, 				   kernel_initializer="glorot_normal", recurrent_initializer="glorot_normal", activation='sigmoid')))
    #model.add(Dropout(0.2))
    
    model.add(Bidirectional(LSTM(units=hidden_size,input_shape=(time_steps,hidden_size),return_sequences=False, kernel_initializer="glorot_normal", recurrent_initializer="glorot_normal", activation='sigmoid')))
    model.add(Dropout(0.2))
    model.add(Dense(no_tags))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', keras_metrics.precision(label=class_value), keras_metrics.recall(label=class_value), keras_metrics.f1_score(label=class_value)])

    return model




model = lstm_model(k_max,64,y_train.shape[1])
print(model.summary())

tbCallBack = TensorBoard(log_dir='data/{}'.format(time()))

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=60, batch_size=1, callbacks=[tbCallBack])

#Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy of LSTM is: %.2f%%" % (scores[1]*100))
print(y_test)



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

'''
plt.plot(history.history['precision'])
plt.plot(history.history['recall'])
plt.title('precision recall curve')
plt.ylabel('precision')
plt.xlabel('epoch')
plt.legend(['precision', 'recall'], loc='upper left') 
plt.show()


plt.plot(history.history['val_precision'])
plt.plot(history.history['val_recall'])
plt.title('precision recall curve')
plt.ylabel('precision')
plt.xlabel('epoch')
plt.legend(['val precision', 'val recall'], loc='upper left') 
plt.show()

plt.plot(history.history['val_precision'])
plt.plot(history.history['val_recall'])
plt.title('precision recall curve')
plt.ylabel('precision')
plt.xlabel('epoch')
plt.legend(['val precision', 'val recall'], loc='upper left') 
plt.show()


plt.plot(history.history['precision'])
plt.plot(history.history['recall'])
plt.plot(history.history['val_precision'])
plt.plot(history.history['val_recall'])
plt.title('precision recall curve')
plt.ylabel('precision')
plt.xlabel('epoch')
plt.legend(['precision', 'recall', 'val precision', 'val recall'], loc='upper left') 
plt.show()'''




# serialize model to YAML
model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
