from HELPER_FUNCTION import save_topology_in_csv, get_topology_only

import random
import csv
import os
import pandas as pd

'''
Convolution layers: num.output_filter, kernel_size, stride
c_1 = 32,3,1
c_2 = 32,4,1
c_3 = 32,5,1
c_4 = 36,3,1
c_5 = 36,4,1
c_6 = 36,5,1
c_7 = 48,3,1
c_8 = 48,4,1
c_9 = 48,5,1
c_10 = 64,3,1
c_11 = 64,4,1
c_12 = 64,5,1

Max pooling layers: kernel , stride
m_1 = 2,2
m_2 = 3,2
m_3 = 5,3

Softmax
s
'''

dict_element = {}
dict_element['c_1'] = [32,3,1]
dict_element['c_2'] = [32,4,1]
dict_element['c_3'] = [32,5,1]
dict_element['c_4'] = [36,3,1]
dict_element['c_5'] = [36,4,1]
dict_element['c_6'] = [36,5,1]
dict_element['c_7'] = [48,3,1]
dict_element['c_8'] = [48,4,1]
dict_element['c_9'] = [48,5,1]
dict_element['c_10'] = [64,3,1]
dict_element['c_11'] = [64,4,1]
dict_element['c_12'] = [64,5,1]
dict_element['m_1'] = [2,2]
dict_element['m_2'] = [3,2]
dict_element['m_3'] = [5,3]
dict_element['s'] = [1]

MAX_NUM_LAYER = 4
LAYER_EMPTY = '-'
LAYER_SOFTMAX = 's'

KEY_MODEL = 'Model'
KEY_FIRST_LAYER = '1st Layer'
KEY_SECOND_LAYER = '2nd Layer'
KEY_THIRD_LAYER = '3rd Layer'
KEY_FORTH_LAYER = '4th Layer'

def have_duplicate(dict_model,temp_list):
    for key,value in dict_model.items():
        if temp_list == value:
            return True
    return False

def count_non_duplicate(dict_model):
    temp_list = []
    for key,value in dict_model.items():
        temp_value = ''.join(value)
        temp_list.append(temp_value)
    return len(set(temp_list))

def fix_topology(dict_model):
    list_of_dict = []
    for key,value in dict_model.items():
        temp_dict = {}
        temp_dict[KEY_MODEL] = key

        temp_dict[KEY_FIRST_LAYER] = value[0]
        if len(value) < 2:
            temp_dict[KEY_SECOND_LAYER] = LAYER_EMPTY
            temp_dict[KEY_THIRD_LAYER] = LAYER_EMPTY
            temp_dict[KEY_FORTH_LAYER] = LAYER_EMPTY
            list_of_dict.append(temp_dict)
            continue

        temp_dict[KEY_SECOND_LAYER] = value[1]
        if len(value) < 3:
            temp_dict[KEY_THIRD_LAYER] = LAYER_EMPTY
            temp_dict[KEY_FORTH_LAYER] = LAYER_EMPTY
            list_of_dict.append(temp_dict)
            continue

        temp_dict[KEY_THIRD_LAYER] = value[2]
        if len(value) < 4:
            temp_dict[KEY_FORTH_LAYER] = LAYER_EMPTY
            list_of_dict.append(temp_dict)
            continue

        temp_dict[KEY_FORTH_LAYER] = value[3]

        list_of_dict.append(temp_dict)
    return list_of_dict

def add_layer(number_model):
    num_model = 0
    dict_model = {}

    while num_model < number_model:
        temp_list = []
        for num_layer in range(MAX_NUM_LAYER):

            element = random.choice(list(dict_element))

            if num_layer == 0:
                if element == LAYER_SOFTMAX or element == 'm_1' or element == 'm_2' or element == 'm_3' :
                    continue
            if num_layer == 1:
                if element == LAYER_SOFTMAX:
                    continue
            temp_list.append(element)

            if element == LAYER_SOFTMAX:
                break
        if len(temp_list) < MAX_NUM_LAYER and temp_list[-1] != LAYER_SOFTMAX:
            temp_list.append(random.choice(list(dict_element)))

        if have_duplicate(dict_model,temp_list):
            continue

        dict_model["model_"+str(num_model+1)] = temp_list
        num_model += 1
    return dict_model

def implement_topology(number_model):
    dict_model = add_layer(number_model)
    num_non_dup = count_non_duplicate(dict_model)
    list_of_dict = fix_topology(dict_model)
    return list_of_dict

# TO ADD: LAST COLUMN OF FILE, use this function
def fn_to_add_column(file_name, content, column_name):
    csv_input = pd.read_csv(file_name)
    csv_input[column_name] =  content
    csv_input.to_csv(file_name, index= False)

def convert_dict_to_list(dict_data):
    list_of_model = []
    for indi_list in dict_data:
        temp_list = [indi_list[KEY_MODEL],indi_list[KEY_FIRST_LAYER], indi_list[KEY_SECOND_LAYER],
                    indi_list[KEY_THIRD_LAYER], indi_list[KEY_FORTH_LAYER],'Unknown','Unknown']
        list_of_model.append(temp_list)
    return list_of_model

def fix_dict(dict_data):
    final_array = []
    data  = convert_dict_to_list(dict_data)
    for array in data:
        temp_array = array[:]
        array = get_topology_only(array)
        count = 0
        for single_element in array:
            if single_element != LAYER_EMPTY:
                count += 1
        if count < MAX_NUM_LAYER and array[count-1] != LAYER_SOFTMAX:
            array[count] = LAYER_SOFTMAX
            temp_array[count+1] = LAYER_SOFTMAX
        final_array.append(temp_array)
    return final_array

def get_random_topology(num_model, output_file_name):
    data_dict = implement_topology(num_model)
    data_dict = fix_dict(data_dict)
    output_file_name= save_topology_in_csv(output_file_name,data_dict)
    print("\nCreate file of random topology named: ",output_file_name, " sucessfull!\n")

    return output_file_name

def main():
    OUTPUT_FILE_NAME  = 'test_random_topology.csv'
    NUMBER_MODEL = 15
    file_name = get_random_topology(NUMBER_MODEL, OUTPUT_FILE_NAME)
    print(file_name)

if __name__ == "__main__":
    main()
