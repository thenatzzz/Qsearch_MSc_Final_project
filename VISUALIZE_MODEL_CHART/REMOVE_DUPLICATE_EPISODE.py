import numpy as np
import sys
sys.path.insert(0,'/homes/nj2217/FINAL_PROJECT/MAIN')

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
    FOLDER = 'FINISHED_MODEL'

    DATASET_FOLDER = 'MNIST'
    # MAIN_FOLDER = 'MNIST_1' # does not have episode_table
    # MAIN_FOLDER = 'MNIST_2'
    # MAIN_FOLDER = 'MNIST_3'
    # MAIN_FOLDER = 'MNIST_4'

    DATASET = 'CIFAR-10'
    # MAIN_FOLDER = 'CIFAR-10_1'   # does not have episode_table
    # MAIN_FOLDER = 'CIFAR-10_2'   # does not have episode_table
    # MAIN_FOLDER = 'CIFAR-10_3'    # does not have episode_table
    # MAIN_FOLDER = 'CIFAR-10_4'    # does not have episode_table
    # MAIN_FOLDER = 'CIFAR-10_5'

    EPISODE_FOLDER = 'EPISODE_TABLE'
    INPUT_FILE = 'original_episode_table.csv'
    OUTPUT_FILE = 'episode_table.csv'

    PATH_DATASET = os.path.join(CURRENT_WORKING_DIR,FOLDER,DATASET_FOLDER,\
                                MAIN_FOLDER,EPISODE_FOLDER)
    INPUT_FILE_WITH_PATH = os.path.join(PATH_DATASET,INPUT_FILE)

    data = get_data_from_csv(INPUT_FILE_WITH_PATH)
    header = data[0]
    data = format_data_without_header(data)

    print("Length before: ",len(data))
    data =remove_repeat_eps(data)
    print("Length after: ",len(data))

    final_data = [header] + data
    OUTPUT_FILE_WITH_PATH = os.path.join(PATH_DATASET,OUTPUT_FILE)
    file = save_list_csv_rowbyrow(OUTPUT_FILE_WITH_PATH,final_data)

if __name__ == '__main__':
    main()
