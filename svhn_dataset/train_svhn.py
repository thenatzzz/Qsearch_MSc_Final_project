from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras
import tempfile
import scipy
import os
import numpy as np
import pandas as pd
from scipy import io
EPOCHS = 1

train_data = scipy.io.loadmat('/vol/bitbucket/nj2217/SVHN/train_32x32.mat')
test_data = scipy.io.loadmat('/vol/bitbucket/nj2217/SVHN/test_32x32.mat')
'''
# SHAPE
y_train:  (73257, 10)
y_test:  (26032, 10)
X_train before:  (73257, 32, 32, 3)
X_train after:  (73257, 32, 32, 3)
X_test before:  (26032, 32, 32, 3)
X_test after:  (26032, 32, 32, 3)
'''
y_train = keras.utils.to_categorical(train_data['y'][:,0])[:,1:]
print("y_train: ", np.shape(y_train))
y_test = keras.utils.to_categorical(test_data['y'][:,0])[:,1:]
print("y_test: ", np.shape(y_test))

X_train = np.zeros((73257, 32, 32, 3))
print("X_train before: ", np.shape(X_train))
for i in range(len(X_train)):
    X_train[i] = train_data['X'].T[i].T.astype('float32')/255
print("X_train after: ", np.shape(X_train))

X_test = np.zeros((26032, 32, 32, 3))
print("X_test before: ", np.shape(X_test))
for i in range(len(X_test)):
    X_test[i] = test_data['X'].T[i].T.astype('float32')/255
print("X_test after: ", np.shape(X_test))

num_classes = 10
cnn = Sequential()
cnn.add(Conv2D(32, kernel_size=(4, 4),
                 activation='relu',
                 input_shape=(32, 32, 3)))
cnn.add(Dropout(0.2))
cnn.add(BatchNormalization())
cnn.add(Conv2D(32, kernel_size = (4,4), activation = 'relu'))
cnn.add(Dropout(0.2))
cnn.add(Conv2D(32, kernel_size = (4,4), activation = 'relu'))
cnn.add(Dropout(0.2))
# cnn.add(BatchNormalization)
cnn.add(MaxPooling2D(pool_size = (2,2)))
cnn.add(Flatten())
cnn.add(Dense(128, activation = 'relu'))
cnn.add(Dense(128, activation = 'relu'))
cnn.add(Dense(num_classes, activation='softmax'))

cnn.compile("adam", "categorical_crossentropy", metrics=['accuracy'])

history_cnn = cnn.fit(X_train, y_train,
                          batch_size=128, epochs=EPOCHS, verbose=1, validation_split=.1)

scores = cnn.evaluate(X_test, y_test, batch_size = 128, verbose = 0)
print("%s: %.2f%%" % (cnn.metrics_names[1], scores[1]*100))
