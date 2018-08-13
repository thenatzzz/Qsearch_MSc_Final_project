import numpy as np
import sys

sys.path.insert(0,'/homes/nj2217/FINAL_PROJECT/MAIN')
from HELPER_FUNCTION import *


def list_file_in_path(path):
    array_file_name = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
    return array_file_name

def main():
    CURRENT_WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
    FOLDER = 'FINISHED_MODEL'
    DATASET_FOLDER = 'MNIST'
    MAIN_FOLDER = 'MNIST_1'
    QTABLE_FOLDER= 'QTABLE'
    PATH_DATASET = os.path.join(CURRENT_WORKING_DIR,FOLDER,DATASET_FOLDER,\
                                MAIN_FOLDER,QTABLE_FOLDER)
    QTABLE_FILE = list_file_in_path(PATH_DATASET)
    print(QTABLE_FILE)

    
if "__main__" == __name__:
    main()
