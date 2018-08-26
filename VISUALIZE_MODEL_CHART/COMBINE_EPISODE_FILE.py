import numpy as np
import sys

sys.path.insert(0,'/homes/nj2217/FINAL_PROJECT/MAIN')
from HELPER_FUNCTION import *

def list_file_in_path(path):
    ############################################################################
    # FUNCTION DESCRIPTION: return all file in the specified path in one array
    ############################################################################
    array_file_name = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
    return array_file_name

def join_path_to_file(FILE_LIST, PATH_DATASET):
    ############################################################################
    # FUNCTION DESCRIPTION: join specifed path to specifed file
    ############################################################################
    if isinstance(FILE_LIST,list):
        for i in range(len(FILE_LIST)):
            FILE_LIST[i] = os.path.join(PATH_DATASET,FILE_LIST[i])

    if isinstance(FILE_LIST, str):
        FILE_LIST = os.path.join(PATH_DATASET,FILE_LIST)

    return FILE_LIST

def main():
    CURRENT_WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
    FOLDER = 'FINISHED_MODEL'

    DATASET_FOLDER = 'MNIST'
    MAIN_FOLDER = 'MNIST_1'

    DATASET_FOLDER = 'CIFAR-10'
    MAIN_FOLDER = 'CIFAR-10_3'

    QTABLE_FOLDER= 'QTABLE'
    PATH_DATASET = os.path.join(CURRENT_WORKING_DIR,FOLDER,DATASET_FOLDER,\
                                MAIN_FOLDER,QTABLE_FOLDER)
    QTABLE_FILE = list_file_in_path(PATH_DATASET)
    QTABLE_FILE_WITH_PATH = join_path_to_file(QTABLE_FILE,PATH_DATASET)

    data_list = []
    for i in range(len(QTABLE_FILE_WITH_PATH)):
        data = get_data_from_csv(QTABLE_FILE_WITH_PATH[i])
        data_list = data_list + data

    OUTPUT_FILE = 'q_table.csv'
    OUTPUT_FILE_WITH_PATH = join_path_to_file(OUTPUT_FILE,PATH_DATASET)
    OUTPUT_FILE = save_list_csv_rowbyrow(OUTPUT_FILE_WITH_PATH,data_list)

    print("Combine : \n\n{} \n\ninto {} successfully!".format(QTABLE_FILE_WITH_PATH,OUTPUT_FILE))

if "__main__" == __name__:
    main()
