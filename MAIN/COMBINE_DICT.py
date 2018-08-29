from RANDOM_TOPOLOGY import fix_dict
from HELPER_FUNCTION import save_topology_in_csv,get_data_from_csv,format_data_without_header
from operator import itemgetter
import os
import pathlib
import numpy as np

LAYER_SOFTMAX = 's'
MAX_NUM_LAYER = 4

def list_file_in_path(path):
    ############################################################################
    # FUNCTION DESCRIPTION: function to list all files in specified path
    ############################################################################
    array_file_name = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
    return array_file_name

def add_path_to_file(path,file):
    ############################################################################
    # FUNCTION DESCRIPTION: function to add '/' to file
    ############################################################################
    file_with_path = path + "/"+file
    return file_with_path

def combine(path,array_file):
    ############################################################################
    # FUNCTION DESCRIPTION: function to combine all file in path into one array
    ############################################################################
    final_list_data = []
    for file in array_file:
        file_with_path = add_path_to_file(path, file)
        data = get_data_from_csv(file_with_path)
        data = format_data_without_header(data)
        final_list_data += data
    return final_list_data

def fix_list(data_list):
    ############################################################################
    # FUNCTION DESCRIPTION: function to fix incorrect data list
    ############################################################################
    final_array = []
    num_model_fixed = 0
    for array in data_list:
        temp_array = array[:]
        array = array[1:-2]
        count = 0
        for single_element in array:
            if single_element != '-':
                count += 1
        if count < MAX_NUM_LAYER and array[count-1] != LAYER_SOFTMAX:
            array[count] = LAYER_SOFTMAX
            temp_array[count+1] = LAYER_SOFTMAX
            num_model_fixed += 1
        final_array.append(temp_array)
    return final_array

def check_dup(data, final_list):
    ############################################################################
    # FUNCTION DESCRIPTION: function to check whether there are duplicate models
    #                       or not
    ############################################################################
    model_name = data[0]
    for data_in_final in final_list:
        model_name_final_list = data_in_final[0]
        if model_name == model_name_final_list:
            return True
    return False

def delete_duplicate(data_list):
    ############################################################################
    # FUNCTION DESCRIPTION: function to remove duplicate data
    ############################################################################
    final_list = []
    num = 0
    for data in data_list:
        is_duplicate = check_dup(data,final_list)

        if is_duplicate:
            continue
        else:
            final_list.append(data)
    return final_list

def strip_model_tag(data):
    ############################################################################
    # FUNCTION DESCRIPTION: function to strip off 'model'tag
    ############################################################################
    temp_model_num = data[0].strip('model_')
    return int(temp_model_num)

def strip_model_list(data_list):
    ############################################################################
    # FUNCTION DESCRIPTION: function to strip off 'model'tag from all given list
    #                       of data
    ############################################################################
    model_index = 0
    for data in data_list:
        data[model_index] = strip_model_tag(data)
    return data_list

def sort_data(data_list, index_to_sort):
    ############################################################################
    # FUNCTION DESCRIPTION: function to sort data according to specified index
    ############################################################################
    data_list = sorted(data_list, key = itemgetter(index_to_sort),reverse = True)
    return data_list

def sort_data_by_name(data_list):
    ############################################################################
    # FUNCTION DESCRIPTION: function to sort model according to name
    ############################################################################
    data_list = strip_model_list(data_list)
    model_name_index = 0
    data_list = sort_data(data_list, model_name_index )
    return data_list

def attach_model_tag(data_list):
    ############################################################################
    # FUNCTION DESCRIPTION: function to attach tag 'model' to model
    ############################################################################
    for data in data_list:
        data[0] = "model_"+str(data[0])
    return data_list

def fix_layer(data_list):
    ############################################################################
    # FUNCTION DESCRIPTION: function to fix layer name
    ############################################################################
    LETTER_INDEX  =0
    for data in data_list:
        for index in range(1,5):
            data[index] = data[index][LETTER_INDEX].lower()+data[index][LETTER_INDEX+1:]
    return data_list

def clean_model_dict(data_list):
    ############################################################################
    # FUNCTION DESCRIPTION: main function to fix data
    ############################################################################
    modified_data = fix_list(data_list)
    modified_data = delete_duplicate(modified_data)
    modified_data = sort_data_by_name(modified_data)
    modified_data = attach_model_tag(modified_data)
    modified_data = fix_layer(modified_data)
    return modified_data

def combine_file(path,output_file):
    ############################################################################
    # FUNCTION DESCRIPTION: function to combine fixed output file to path
    ############################################################################
    array_files = list_file_in_path(path)
    data = combine(path,array_files)
    data = clean_model_dict(data)
    return data

def remove_data_initial_ep(data,initial_episode):
    ############################################################################
    # FUNCTION DESCRIPTION: function to get data after initial episode
    ############################################################################
    return data[initial_episode+1:]

def get_top_model(sorted_data,num_model):
    ############################################################################
    # FUNCTION DESCRIPTION: function to get top model with specified number of model
    ############################################################################
    temp_list = []
    for i in range(num_model):
        temp_list.append(sorted_data[i])
    return np.asarray(temp_list)

def get_mean_initial_ep(data,initial_episode,index_target):
    ############################################################################
    # FUNCTION DESCRIPTION: function to get average of accuracy or loss of specified
    #                       initial episodes
    ############################################################################
    sum = 0
    for i in range(initial_episode):
        sum += float(data[i][index_target])
    return sum/initial_episode

def main():

    CURRENT_DIR = os.getcwd()

    INPUT_FOLDER_NAME = 'CIFAR_DICT'
    INPUT_PATH = CURRENT_DIR +'/'+INPUT_FOLDER_NAME

    OUTPUT_FILE_NAME = "COMPLETE_DICT.csv"
    OUTPUT_FILE_PATH = CURRENT_DIR + '/'+ OUTPUT_FILE_NAME
    INDEX_ACCURACY = -2
    INDEX_LOSS = -1
    SORTING_INDEX = INDEX_ACCURACY

    file_path = "/homes/nj2217/FINAL_PROJECT/MAIN/FINISHED_MODEL/"

    '''
    file = file_path + file_type + main_folder + main_file
    data = get_data_from_csv(file)
    data = format_data_without_header(data)
    # sorted_data = sort_data(data,SORTING_INDEX)
    print("\nLast model is : ", data[-1])
    file = file.strip('.csv')
    file += "_sorted.csv"
    # result_file_name = save_topology_in_csv(file,sorted_data)  #Uncomment to save new data in file

    initial_eps_mean = get_mean_initial_ep(data,initial_episode,SORTING_INDEX)
    print("\nMean of initial: ",initial_episode, " is --> ", initial_eps_mean)

    data_without_initial_ep = remove_data_initial_ep(data,initial_episode)
    sorted_data = sort_data(data_without_initial_ep,SORTING_INDEX)
    num_model = 5
    list_top_model = get_top_model(sorted_data,num_model)
    print("\nThe list of top ",num_model," are as follows:\n ",list_top_model)
    '''

    CURRENT_WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
    '''Model with Random Search'''
    folder = "FINISHED_MODEL"
    file_type = "MODEL_WITH_RANDOM_SEARCH/"

    main_folder = "CIFAR-10/"
    # main_folder = "MNIST/"

    main_file = "original_model.csv"
    # file = file_path + file_type+ main_folder+ main_file
    file = os.path.join(CURRENT_WORKING_DIR,folder,file_type,main_folder,main_file)
    data = get_data_from_csv(file)
    data =format_data_without_header(data)
    sorted_data = sort_data(data,SORTING_INDEX)
    num_model = 5
    list_top_model = get_top_model(sorted_data,num_model)
    print("\nThe list of top ",num_model," are as follows:\n ",list_top_model)

if __name__ == "__main__":
    main()
