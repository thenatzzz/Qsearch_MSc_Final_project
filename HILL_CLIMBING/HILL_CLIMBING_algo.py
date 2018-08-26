import numpy as np

import sys
sys.path.insert(0,'/homes/nj2217/FINAL_PROJECT/MAIN')
# sys.path.insert(0,'/d/PROJECT/FINAL_PROJECT/MAIN')

from HELPER_FUNCTION import *
from TRAIN_MODEL_CIFAR10 import *
from TRAIN_MODEL_MNIST import *

MAX_ACTION = 16
MAX_STATE = 4
LAYER_ACTION = ['c_1','c_2','c_3','c_4','c_5','c_6','c_7','c_8','c_9','c_10','c_11',\
                'c_12','m_1','m_2','m_3','s']

MAX_INDEX_MODEL_ARRAY = 1 + MAX_STATE + 2
INDEX_MODEL = 0

def create_empty_array():
    final_array = []
    for _ in range(MAX_ACTION*MAX_STATE):
        temp_array = ['-']*MAX_INDEX_MODEL_ARRAY
        final_array.append(temp_array)
    return final_array

def init_layer_action(final_array,index_hill_level):
    for index in range(MAX_ACTION):
        final_array[index+index_hill_level*MAX_ACTION][index_hill_level+1] = LAYER_ACTION[index]
    return final_array

def hill_climbing(DATASET):
    model_num = 0
    final_array = create_empty_array()

    for index_hill_level in range(MAX_STATE):
        final_array = init_layer_action(final_array,index_hill_level)
        for index_layer_in_hill in range(MAX_ACTION):
            for index_model_array in range(MAX_INDEX_MODEL_ARRAY):
                final_array[index_layer_in_hill+index_hill_level*MAX_ACTION][INDEX_MODEL] = 'model_'+str(model_num)

                if DATASET == 'cifar10':

                elif DATASET == 'mnist':
            model_num += 1

    print(final_array)
def main():
    DATASET = 'cifar10'
    hill_climbing(DATASET)

if __name__ == '__main__':
    main()
