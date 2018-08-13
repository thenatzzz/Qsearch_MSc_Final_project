'''Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.backend import clear_session
from keras.layers import BatchNormalization

import numpy as np
import csv
import os
import pandas as pd
import scipy
import tempfile

from scipy import io

batch_size = 32
num_classes = 10
epochs = 20
data_augmentation = True
# data_augmentation = False

num_predictions = 20
saved_model_path = "/vol/bitbucket/nj2217/CIFAR-10/"
save_dir = os.path.join(saved_model_path, 'tmp4_verified_cifar10_models')
# save_dir = os.path.join(saved_model_path, 'verified_cifar10_models2')


# save_dir = os.path.join(saved_model_path, 'saved_models')
# model_name = 'keras_cifar10_trained_model.h5'

#PREDEFINED LAYER
# 1. Convolutional Layer [number of output filter, kernel size, stride]
c_1 = [32,3,1]
c_2 = [32,4,1]
c_3 = [32,5,1]
c_4 = [36,3,1]
c_5 = [36,4,1]
c_6 = [36,5,1]
c_7 = [48,3,1]
c_8 = [48,4,1]
c_9 = [48,5,1]
c_10 = [64,3,1]
c_11 = [64,4,1]
c_12 = [64,5,1]
# 2. Pooling Layer [kernel size, stride]
m_1 = [2,2]
m_2 = [3,2]
m_3 = [5,3]
# 3. Softmax Layer (Termination Layer)
s = [0]

def add_conv2D(model, layer_param):
    num_filters = layer_param[0]
    size_kernel = layer_param[1]
    num_stride = layer_param[2]
    model.add(Conv2D(num_filters, kernel_size=(size_kernel, size_kernel),border_mode='same', activation='relu'))

def add_maxpool2D(model, layer_param):
    size_kernel = layer_param[0]
    num_stride = layer_param[1]
    model.add(MaxPooling2D(pool_size=(size_kernel, size_kernel),strides= (num_stride,num_stride),border_mode='same' ))
    model.add(Dropout(0.25))

def count_model_layer(model_from_csv):
    print("______________________________________")
    count = 0
    print(model_from_csv, " :::::::::::::::")
    for i in range(len(model_from_csv)):
        count += 1
        if model_from_csv[i] == 's':
            break
    return count

def open_file(file_csv):
    with open(file_csv) as f:
        reader = csv.reader(f, delimiter=',' , quotechar= ',' ,
                        quoting = csv.QUOTE_MINIMAL)
        data  = [r for r in reader]
        data = data[1:]      # To get rid of column name
    return data

def train_model(model_from_csv):
    for i in range(len(model_from_csv)):
        print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHh")
        print(model_from_csv[i][1:-2])
        print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")

        '''
        # The data, split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # Convert class vectors to binary class matrices.
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        '''
        train_data = scipy.io.loadmat('/vol/bitbucket/nj2217/SVHN/train_32x32.mat')
        test_data = scipy.io.loadmat('/vol/bitbucket/nj2217/SVHN/test_32x32.mat')
        y_train = keras.utils.to_categorical(train_data['y'][:,0])[:,1:]
        print("y_train: ", np.shape(y_train))
        y_test = keras.utils.to_categorical(test_data['y'][:,0])[:,1:]
        print("y_test: ", np.shape(y_test))

        X_train = np.zeros((73257, 32, 32, 3))
        print("X_train before: ", np.shape(X_train))
        for index in range(len(X_train)):
            X_train[index] = train_data['X'].T[index].T.astype('float32')/255
        print("X_train after: ", np.shape(X_train))

        X_test = np.zeros((26032, 32, 32, 3))
        print("X_test before: ", np.shape(X_test))
        for index in range(len(X_test)):
            X_test[index] = test_data['X'].T[index].T.astype('float32')/255
        print("X_test after: ", np.shape(X_test))


        model = Sequential()
        print(model_from_csv)
        print("i: ", i)
        print(model_from_csv[i])
        print(model_from_csv[i][:])
        num_layer = count_model_layer(model_from_csv[i][1:-2])
        print("num_layer: ",num_layer)
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))

        for index in range(1,num_layer+1):
            print("index inside train_model: ",index)
            print("layer trained : ", model_from_csv[i][index])
            if model_from_csv[i][index] == 'c_1':
                print("inside c_1")
                add_conv2D(model,c_1)
            elif model_from_csv[i][index] == 'c_2':
                add_conv2D(model,c_2)
            elif model_from_csv[i][index] == 'c_3':
                add_conv2D(model,c_3)
            elif model_from_csv[i][index] == 'c_4':
                add_conv2D(model,c_4)
            elif model_from_csv[i][index] == 'c_5':
                add_conv2D(model,c_5)
            elif model_from_csv[i][index] == 'c_6':
                add_conv2D(model,c_6)
            elif model_from_csv[i][index] == 'c_7':
                add_conv2D(model,c_7)
            elif model_from_csv[i][index] == 'c_8':
                add_conv2D(model,c_8)
            elif model_from_csv[i][index] == 'c_9':
                add_conv2D(model,c_9)
            elif model_from_csv[i][index] == 'c_10':
                add_conv2D(model,c_10)
            elif model_from_csv[i][index] == 'c_11':
                add_conv2D(model,c_11)
            elif model_from_csv[i][index] == 'c_12':
                add_conv2D(model,c_12)
            elif model_from_csv[i][index] == 'm_1':
                add_maxpool2D(model,m_1)
            elif model_from_csv[i][index] == 'm_2':
                add_maxpool2D(model,m_2)
            elif model_from_csv[i][index] == 'm_3':
                add_maxpool2D(model,m_3)
            elif model_from_csv[i][index] == 's':
                break

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        # initiate RMSprop optimizer
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        # Let's train the model using RMSprop
        model.compile(loss='categorical_crossentropy',
                        optimizer=opt,
                        metrics=['accuracy'])

        x_train = X_train
        x_test = X_test


        # x_train = x_train.astype('float32')
        # x_test = x_test.astype('float32')
        # x_train /= 255
        # x_test /= 255

        if not data_augmentation:
            print('Not using data augmentation.')
            model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    shuffle=True)
        else:
            print('Using real-time data augmentation.')
            # This will do preprocessing and realtime data augmentation:
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                zca_epsilon=1e-06,  # epsilon for ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                shear_range=0.,  # set range for random shear
                zoom_range=0.,  # set range for random zoom
                channel_shift_range=0.,  # set range for random channel shifts
                fill_mode='nearest',  # set mode for filling points outside the input boundaries
                cval=0.,  # value used for fill_mode = "constant"
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False,  # randomly flip images
                rescale=None,  # set rescaling factor (applied before any other transformation)
                preprocessing_function=None,  # set function that will be applied on each input
                data_format=None,  # image data format, either "channels_first" or "channels_last"
                validation_split=0.0)  # fraction of images reserved for validation (strictly between 0 and 1)

            # Compute quantities required for feature-wise normalization
            # (std, mean, and principal components if ZCA whitening is applied).
            datagen.fit(x_train)

            # Fit the model on the batches generated by datagen.flow().
            model.fit_generator(datagen.flow(x_train, y_train,
                                 batch_size=batch_size),
                                 epochs=epochs,
                                 validation_data=(x_test, y_test),
                                 callbacks=[EarlyStopping(min_delta=0.001, patience=3)],
                                 workers=4)
            # model_name = 'keras_cifar10_trained_model.h5'
            model_name = 'keras_cifar10_'
            model_name = model_name + model_from_csv[i][0] + '.h5'
             # Save model and weights
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            model_path = os.path.join(save_dir, model_name)
            model.save(model_path)
            print('Saved trained model at %s ' % model_path)

            # Score trained model.
            scores = model.evaluate(x_test, y_test, verbose=1)
            print('Test loss:', scores[0])
            print('Test accuracy:', scores[1])

            loss = scores[0]
            accuracy = scores[1]
            print("Model ", model_from_csv[i][:-2])
            #save_file(model_from_csv[i],loss,accuracy)
            print("MODEL ",model_from_csv[i][0], " finishes training!")
            print("Number of model trained: ",i+1)
            print('\n')
            clear_session()

if __name__ == '__main__':
    # file_name = "model1201_1500.csv"

    # data = open_file(file_name)
    # FIRST_LAYER = 'c_1'
    # SECOND_LAYER = 'c_7'
    # THIRD_LAYER ='c_11'
    # FORTH_LAYER = 'c_9'
    FIRST_LAYER = 'c_2'
    SECOND_LAYER = 'c_9'
    THIRD_LAYER ='c_11'
    FORTH_LAYER = 'c_1'

    temp = [FIRST_LAYER, SECOND_LAYER,THIRD_LAYER, FORTH_LAYER]
    last_layer = ["unknown", "unknown"]
    data = [["verified_model"]+temp + last_layer]
    train_model(data)
