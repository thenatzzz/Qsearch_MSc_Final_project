from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from HELPER_FUNCTION import format_data_without_header,get_data_from_csv, \
                            get_topology_only, check_complete_model,\
                            count_model_layer, get_latest_model_list,\
                            get_current_model_number, get_new_model_number,\
                            save_trained_model_in_csv
import numpy as np
import tensorflow as tf
import csv
import os
import pandas as pd
tf.logging.set_verbosity(tf.logging.INFO)

MAIN_FILE = "fixed_model_dict.csv"
BATCH_SIZE = 100
TRAINING_STEPS = 10000

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

GLOBAL_DATA = ""
INDEX_MODEL = 0

def make_conv2d(input_layer,layer_param):
    num_filters = layer_param[0]
    size_kernel = layer_param[1]
    num_stride = layer_param[2]
    return tf.layers.conv2d(
            inputs = input_layer,
            filters = num_filters,
            kernel_size = [size_kernel,size_kernel],
            padding = "same",
            activation= tf.nn.relu)

def make_pool2d(input_layer,layer_param):
    size_kernel = layer_param[0]
    num_stride = layer_param[1]
    return tf.layers.max_pooling2d(inputs= input_layer,
                                    pool_size=[size_kernel,size_kernel],
                                    strides=num_stride,
                                    padding= "SAME")

def cnn_model_fn_2(features,labels, mode):

    tmp_single_model = get_topology_only(GLOBAL_DATA)
    num_layer = count_model_layer(tmp_single_model)
    input_layer = tf.reshape(features["x"], [-1,28,28,1])

    layer = input_layer
    temp_layer = 0
    for index  in range(1,num_layer):

        if GLOBAL_DATA[index]== 'c_1':
            temp_layer = make_conv2d(layer, c_1)
        elif GLOBAL_DATA[index] == 'c_2':
            temp_layer = make_conv2d(layer, c_2)
        elif GLOBAL_DATA[index] == 'c_3':
            temp_layer = make_conv2d(layer, c_3)
        elif GLOBAL_DATA[index] == 'c_4':
            temp_layer = make_conv2d(layer, c_4)
        elif GLOBAL_DATA[index] == 'c_5':
            temp_layer = make_conv2d(layer, c_5)
        elif GLOBAL_DATA[index] == 'c_6':
            temp_layer = make_conv2d(layer, c_6)
        elif GLOBAL_DATA[index] == 'c_7':
            temp_layer = make_conv2d(layer, c_7)
        elif GLOBAL_DATA[index] == 'c_8':
            temp_layer = make_conv2d(layer, c_8)
        elif GLOBAL_DATA[index] == 'c_9':
            temp_layer = make_conv2d(layer, c_9)
        elif GLOBAL_DATA[index] == 'c_10':
            temp_layer = make_conv2d(layer, c_10)
        elif GLOBAL_DATA[index] == 'c_11':
            temp_layer = make_conv2d(layer, c_11)
        elif GLOBAL_DATA[index] == 'c_12':
            temp_layer = make_conv2d(layer, c_12)
        elif GLOBAL_DATA[index] == 'm_1':
            temp_layer = make_pool2d(layer, m_1)
        elif GLOBAL_DATA[index] == 'm_2':
            temp_layer = make_pool2d(layer, m_2)
        elif GLOBAL_DATA[index] == 'm_3':
            temp_layer = make_pool2d(layer, m_3)
        elif GLOBAL_DATA[index] == 's':
            break
        layer = temp_layer

    shape_array = layer.get_shape()
    pool2_flat = tf.reshape(layer, [-1, shape_array[1] * shape_array[2] * shape_array[3]])

    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
      train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def check_format(single_model):
    is_verified = True
    if len(single_model) == 4 :
        return  is_verified, [["verified_model"]+single_model+["Unknown","Unknown"]]
    else:
        return not is_verified, single_model

def load_data_mnist():
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    return mnist, train_data, train_labels, eval_data, eval_labels

def implement_cnn(is_verify = False):
    if not is_verify:
        return tf.estimator.Estimator(model_fn = cnn_model_fn_2, model_dir = \
        "/vol/bitbucket/nj2217/PROJECT_1/mnist_convnet_model"+ "_"+str(INDEX_MODEL))
    else:
        return tf.estimator.Estimator(model_fn = cnn_model_fn_2)

def set_up_logging():
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)
  return logging_hook

def train_the_model(mnist_classifier,train_data,train_labels,logging_hook):
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": train_data},
                    y=train_labels,
                    batch_size=BATCH_SIZE,
                    num_epochs=None,
                    shuffle=True)

    mnist_classifier.train(
                    input_fn=train_input_fn,
                    steps=TRAINING_STEPS,
                    hooks=[logging_hook])

def evaluate_model(mnist_classifier,eval_data,eval_labels):
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
  return mnist_classifier.evaluate(input_fn=eval_input_fn)

def make_data_global(single_model):
    global GLOBAL_DATA
    GLOBAL_DATA = single_model
    return GLOBAL_DATA

def reset_global_data():
    global GLOBAL_DATA
    GLOBAL_DATA = ""

def train_model_mnist(single_model, is_verify = False):
  file = MAIN_FILE

  global INDEX_MODEL
  is_complete_model = check_complete_model(single_model)

  if not is_complete_model:
      single_model = get_latest_model_list(single_model, file)
      model_name = single_model[0]
      cur_model_num = get_current_model_number(model_name)
      INDEX_MODEL = get_new_model_number(cur_model_num)

  temp_single_model = make_data_global(single_model)

  mnist,train_data,train_labels,eval_data,eval_labels= load_data_mnist()
  mnist_classifier = implement_cnn(is_verify)
  logging_hook = set_up_logging()
  train_the_model(mnist_classifier,train_data,train_labels,logging_hook)
  eval_results = evaluate_model(mnist_classifier,eval_data,eval_labels)

  print(eval_results)

  if not is_verify:
      save_trained_model_in_csv(file,temp_single_model,eval_results)

  print(temp_single_model)
  reset_global_data()
  INDEX_MODEL += 1

  return eval_results['accuracy']

def pre_train_model_mnist(file_name,output_file_name):
    global MAIN_FILE
    MAIN_FILE = output_file_name

    data = get_data_from_csv(file_name)
    data = format_data_without_header(data)

    for index in range(len(data)):
        single_model = data[index]
        train_model_mnist(single_model)
