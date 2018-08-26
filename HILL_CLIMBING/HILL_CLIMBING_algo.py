import numpy as np

import sys
sys.path.insert(0,'/homes/nj2217/FINAL_PROJECT/MAIN')

from HELPER_FUNCTION import *

MAX_ACTION = 16
MAX_STATE = 4
LAYER_ACTION = ['c_1','c_2','c_3','c_4','c_5','c_6','c_7','c_8','c_9','c_10','c_11',\
                'c_12','m_1','m_2','m_3','s']

MAX_INDEX_MODEL_ARRAY = 1 + MAX_STATE + 2

def hill_climbing(DATASET):
    for index_hill_level in range(MAX_STATE):
        for index_layer_in_hill in range(MAX_ACTION):
            
            for index_model_array in range(MAX_INDEX_MODEL_ARRAY):


def main():
    DATASET = 'cifar10'
    hill_climbing(DATASET)

if __name__ == '__main__':
    main()
