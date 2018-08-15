import numpy as np
import sys
sys.path.insert(0,'/homes/nj2217/FINAL_PROJECT/MAIN')

from HELPER_FUNCTION import *
INDEX_MODEL_NAME = 0

def fix_model_number(data):
    for i in range(len(data)):
        data[i][INDEX_MODEL_NAME] = 'model_'+str(i)
    return data

def main():

    CURRENT_WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
    FOLDER = 'FINISHED_MODEL'

    DATASET_FOLDER = 'MNIST'
    MAIN_FOLDER = 'MNIST_1' # does not have episode_table
    # MAIN_FOLDER = 'MNIST_2'
    # MAIN_FOLDER = 'MNIST_3'
    # MAIN_FOLDER = 'MNIST_4'

    # DATASET_FOLDER = 'CIFAR-10'
    # MAIN_FOLDER = 'CIFAR-10_1'   # does not have episode_table
    # MAIN_FOLDER = 'CIFAR-10_2'   # does not have episode_table
    # MAIN_FOLDER = 'CIFAR-10_3'    # does not have episode_table
    # MAIN_FOLDER = 'CIFAR-10_4'    # does not have episode_table
    # MAIN_FOLDER = 'CIFAR-10_5'

    MODEL_FOLDER = 'MODEL_DICT'
    INPUT_FILE = 'original_model.csv'
    OUTPUT_FILE = 'original_model_2.csv'

    PATH_DATASET = os.path.join(CURRENT_WORKING_DIR,FOLDER,DATASET_FOLDER,\
                                MAIN_FOLDER,MODEL_FOLDER)
    INPUT_FILE_WITH_PATH = os.path.join(PATH_DATASET,INPUT_FILE)
    data = get_data_from_csv(INPUT_FILE_WITH_PATH)
    header = data[0]
    data = format_data_without_header(data)

    data =fix_model_number(data)

    final_data = [header] + data
    OUTPUT_FILE_WITH_PATH = os.path.join(PATH_DATASET,OUTPUT_FILE)
    file = save_list_csv_rowbyrow(OUTPUT_FILE_WITH_PATH,final_data)

if __name__ == '__main__':
    main()
