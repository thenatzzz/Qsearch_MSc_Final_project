import numpy as np
import sys

sys.path.insert(0,'/homes/nj2217/PROJECT/TMP4_CLEAN_CODE')

from HELPER_FUNCTION import *
INDEX_EPS_NAME = 0

def remove_repeat_eps(data):
    final_list = []
    index_elem = 0
    for datum in data:
        if index_elem != 0 and datum[INDEX_EPS_NAME] == final_list[index_elem-1][INDEX_EPS_NAME]:
            continue
        final_list.append(datum)
        index_elem += 1

    return final_list

def main():
    CURRENT_WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_FOLDER = 'FINISHED_MODEL'

    DATASET = 'MNIST'
    MAIN_FOLDER = 'CLEAN_CODE'
    MAIN_FOLDER = 'MNIST_GPU'
    MAIN_FOLDER = 'MNIST_GPU_UPDATE_EXP'
    # MAIN_FOLDER = 'MNIST_GPU_UPDATE_EXP_PERIOD'

    # DATASET = 'CIFAR-10'
    # MAIN_FOLDER = 'TMP_CODE'
    # MAIN_FOLDER = 'TMP2_CLEAN_CODE'
    # MAIN_FOLDER = 'TMP3_CLEAN_CODE'
    # MAIN_FOLDER = 'TMP_CODE_2'
    # MAIN_FOLDER = 'TMP2_CODE_2'

    CSV_FILE_NAME = 'episode_fixed_model.csv'
    PATH_DATASET = os.path.join(CURRENT_WORKING_DIR,DATASET_FOLDER,DATASET,MAIN_FOLDER,CSV_FILE_NAME)
    data = get_data_from_csv(PATH_DATASET)
    header = data[0]
    data = format_data_without_header(data)
    print("Length before: ",len(data))
    data =remove_repeat_eps(data)
    print("Length after: ",len(data))
    final_data = [header] + data
    file = save_list_csv_rowbyrow('test.csv',final_data)

if __name__ == '__main__':
    main()
