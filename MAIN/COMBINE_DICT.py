from RANDOM_TOPOLOGY import fix_dict
from HELPER_FUNCTION import save_topology_in_csv,get_data_from_csv,format_data_without_header
from operator import itemgetter
import os
import pathlib
import numpy as np

LAYER_SOFTMAX = 's'
MAX_NUM_LAYER = 4

def list_file_in_path(path):
    array_file_name = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
    return array_file_name

def add_path_to_file(path,file):
    file_with_path = path + "/"+file
    return file_with_path

def combine(path,array_file):
    final_list_data = []
    for file in array_file:
        file_with_path = add_path_to_file(path, file)
        data = get_data_from_csv(file_with_path)
        data = format_data_without_header(data)
        final_list_data += data
    return final_list_data

def fix_list(data_list):
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
            # print("Before fixed: ",array)
            array[count] = LAYER_SOFTMAX
            temp_array[count+1] = LAYER_SOFTMAX
            num_model_fixed += 1
            # print("After fixed: ",array)
        final_array.append(temp_array)
    # print(num_model_fixed)
    return final_array

def check_dup(data, final_list):
    model_name = data[0]
    for data_in_final in final_list:
        model_name_final_list = data_in_final[0]
        if model_name == model_name_final_list:
            return True
    return False

def delete_duplicate(data_list):
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
    temp_model_num = data[0].strip('model_')
    return int(temp_model_num)

def strip_model_list(data_list):
    model_index = 0
    for data in data_list:
        data[model_index] = strip_model_tag(data)
    return data_list

def sort_data(data_list, index_to_sort):
    data_list = sorted(data_list, key = itemgetter(index_to_sort),reverse = True)
    return data_list

def sort_data_by_name(data_list):
    data_list = strip_model_list(data_list)
    model_name_index = 0
    data_list = sort_data(data_list, model_name_index )
    return data_list

def attach_model_tag(data_list):
    for data in data_list:
        data[0] = "model_"+str(data[0])
    return data_list

def fix_layer(data_list):
    LETTER_INDEX  =0
    for data in data_list:
        for index in range(1,5):
            data[index] = data[index][LETTER_INDEX].lower()+data[index][LETTER_INDEX+1:]
    return data_list

def clean_model_dict(data_list):
    modified_data = fix_list(data_list)
    modified_data = delete_duplicate(modified_data)
    modified_data = sort_data_by_name(modified_data)
    modified_data = attach_model_tag(modified_data)
    modified_data = fix_layer(modified_data)
    return modified_data

def combine_file(path,output_file):
    array_files = list_file_in_path(path)
    data = combine(path,array_files)
    # print("Before clean-> length data: ", len(data))
    data = clean_model_dict(data)
    # print("After clean-> length_data: ",len(data))
    # print(data)
    return data

def remove_data_initial_ep(data,initial_episode):
    return data[initial_episode+1:]

def get_top_model(sorted_data,num_model):
    temp_list = []
    for i in range(num_model):
        temp_list.append(sorted_data[i])
    return np.asarray(temp_list)

def get_mean_initial_ep(data,initial_episode,index_target):
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
    # SORTING_INDEX = INDEX_LOSS

    #data = combine_file(INPUT_PATH, OUTPUT_FILE_PATH)
    #data = sort_data(data, SORTING_INDEX)

    #result_file_name = save_topology_in_csv(OUTPUT_FILE_PATH,data)
    #print("Combine file from: ",INPUT_PATH, " : and get ", OUTPUT_FILE_NAME, " successfully!")


    file_path = "/homes/nj2217/PROJECT/TMP5_CLEAN_CODE/FINISHED_MODEL/"

    ''' DATASET = mnist '''

  #   '''1 mnist'''
    # c_2,c_6,c_3,c_1
    # total ep: 3500 : 2500/500/500 exp:10
    # file_type = "MNIST/"
    # main_folder = "CLEAN_CODE/"
    # main_file = "mnist_model_dict.csv"
    # initial_episode = 2500
  #   # Average of 2500 model trained at epsilon 1.0: 0.9649
  # #   The list of top  5  are as follows:
  # # [['model_3186' 'c_1' 'c_5' 'c_10' 'c_1' '0.9765' '0.078444734']
  # # ['model_2796' 'c_4' 'c_10' 'c_8' 'm_3' '0.9762' '0.08101152']
  # # ['model_3357' 'c_2' 'c_1' 'c_9' 'm_2' '0.9761' '0.0782957']
  # # ['model_4120' 'c_2' 'c_1' 'c_12' 'c_1' '0.976' '0.07825506']
  # # ['model_3045' 'c_5' 'c_11' 'c_7' 'c_4' '0.9759' '0.08239954']]
  # # Last model is :  ['model_4481', 'm_3', 'c_6', 'm_1', 'c_1', '0.794', '0.7050015']

    '''2 mnist'''
    # c_1,c_6,c_5,m_2
    # total ep: 3800 : 2000/1000/800 exp: 0
    # file_type = "MNIST/"
    # main_folder = "MNIST_GPU/"
    # main_file = "mnist_model_dict.csv"
    # initial_episode = 2000
    # Last model is :  ['model_3101', 'c_1', 'c_6', 'c_1', 'c_10', '0.9726', '0.08008866']
    # Mean of initial:  2000  is -->  0.968884249999998
    # The list of top  5  are as follows:
    # [['model_2996' 'c_5' 'c_2' 'c_8' 'm_2' '0.9766' '0.08379604']
    # ['model_2284' 'c_1' 'c_4' 'c_9' 'm_3' '0.9762' '0.079748444']
    # ['model_2306' 'c_11' 'c_3' 'c_10' 'c_3' '0.9762' '0.08001152']
    # ['model_2589' 'c_1' 'c_2' 'c_2' 'c_8' '0.9762' '0.07942703']
    # ['model_2998' 'c_1' 'c_7' 'c_10' 'c_8' '0.976' '0.08009651']]

    '''3 mnist '''
    # c_8, c_9, c_8, c_9 not decisive
    # total ep: 3800 : 2000/1000/800 exp: 50
    # file_type = "MNIST/"
    # main_folder = "MNIST_GPU_UPDATE_EXP/"
    # main_file = "mnist_model_dict.csv"
    # initial_episode = 2000
    # Last model is :  ['model_3517', 'm_2', 'c_9', 'c_9', 'c_9', '0.9326', '0.22246318']
    # Mean of initial:  2000  is -->  0.968884249999998
    # The list of top  5  are as follows:
    # [['model_2074' 'c_1' 'c_4' 'c_5' 'm_1' '0.9759' '0.08084566']
    # ['model_3389' 'c_4' 'c_2' 'c_12' 'm_1' '0.9759' '0.07729348']
    # ['model_2534' 'c_7' 'c_5' 'c_4' 'm_2' '0.9758' '0.07847521']
    # ['model_2080' 'c_4' 'c_4' 'c_4' 'c_3' '0.9757' '0.080006644']
    # ['model_2352' 'c_7' 'c_10' 'c_10' 'c_8' '0.9757' '0.07875745']]

    '''4 mnist '''
    # c_5, c_8, c_5, c_1
    # total ep: 3800 : 2000/1000/800 exp: 50 periodically
    # file_type = "MNIST/"
    # main_folder = "MNIST_GPU_UPDATE_EXP_PERIOD/"
    # main_file = "mnist_model_dict.csv"
    # initial_episode = 2000
    # Last model is :  ['model_3110', 'c_5', 'c_8', 'c_11', 'm_1', '0.9728', '0.08856994']
    # Mean of initial:  2000  is -->  0.968884249999998
    # The list of top  5  are as follows:
    # [['model_2625' 'c_8' 'c_5' 'c_3' 'c_1' '0.976' '0.08027729']
    #  ['model_2816' 'c_10' 'c_7' 'c_10' 'c_1' '0.976' '0.07784157']
    #  ['model_2547' 'c_2' 'c_1' 'c_5' 'c_3' '0.9756' '0.08090415']
    #  ['model_2612' 'c_5' 'c_8' 'c_5' 'c_8' '0.9755' '0.0773185']
    #  ['model_2760' 'c_5' 'c_3' 'c_5' 'c_1' '0.9755' '0.08051262']]


    ''' DATASET = cifar-10 '''

    file_type = "CIFAR-10/"

    '''1 cifar10'''
    # c_2,c_6,c_11,c_9
    # total ep: 2500 : 1500/500/500 exp:5
    # main_folder = "TMP_CODE/"
    # main_file = "cifar10_model_dict.csv"
    # initial_episode = 1500
    # # Last model is :  ['model_2703', 'c_4', 'c_6', 'c_6', 'c_9', '0.7251', '0.8535752796173096']
    # # Mean of initial:  1500  is -->  0.7221946000000001
    # # The list of top  5  are as follows:
    # #   [['model_2332' 'c_12' 'm_2' 'c_11' 'c_11' '0.7912' '0.6738603033065796']
    # #  ['model_2048' 'c_11' 'm_1' 'c_9' 'c_3' '0.7908' '0.6514748561382294']
    # #  ['model_2141' 'c_11' 'm_3' 'c_11' 'c_12' '0.7884' '0.6525064031124115']
    # #  ['model_2325' 'c_12' 'c_6' 'm_3' 'c_9' '0.7857' '0.6478681169509888']
    # #  ['model_1568' 'c_11' 'c_9' 'm_3' 'c_9' '0.7782' '0.6462198762893677']]

    #'''2 cifar10'''
    # main_folder = "TMP2_CLEAN_CODE/"
    # main_file = ""
    # initial_episode = 1800

    '''2 cifar10'''
    #c_2,c_6,c_11,c_6
    # total ep: 3300 : 2000/800/500 exp:5
    # main_folder = "TMP3_CLEAN_CODE/"
    # main_file = "cifar10_model_dict.csv"
    # initial_episode = 2000
    # # Last model is :  ['model_2979', 'c_2', 'c_6', 'c_5', 'c_9', '0.7234', '0.8119809470176697']
    # # Mean of initial:  2000  is -->  0.7226369500000003
    # # The list of top  5  are as follows:
    # #   [['model_2556' 'c_11' 'm_3' 'c_11' 'c_11' '0.8' '0.6549119873046875']
    # #  ['model_2547' 'c_2' 'c_12' 'm_3' 'c_12' '0.7862' '0.6497823998451233']
    # #  ['model_2678' 'c_2' 'm_3' 'c_11' 'c_11' '0.7822' '0.6641582245826722']
    # #  ['model_2415' 'c_2' 'm_1' 'c_12' 'c_11' '0.7821' '0.6662089567184448']
    # #  ['model_2289' 'c_5' 'm_1' 'c_8' 'c_9' '0.7816' '0.6530621880531311']]

    '''3 cifar10'''
    # c_7,c_6,c_11,c_9
    # total ep: 3800 : 2000/1000/800 exp: 5
    # main_folder = "TMP_CODE_2/"
    # main_file = "cifar10_model_dict.csv"
    # initial_episode = 2000
    # Last model is :  ['model_3407', 'm_3', 'c_4', 'c_11', 'c_9', '0.6897', '0.932465117263794']
    # Mean of initial:  2000  is -->  0.7224285000000003
    # The list of top  5  are as follows:
    #   [['model_3228' 'c_10' 'm_3' 'c_8' 'c_9' '0.7941' '0.6601475175380707']
    #  ['model_2988' 'c_10' 'm_1' 'c_12' 'c_9' '0.7911' '0.6650652590751648']
    #  ['model_2273' 'c_12' 'm_1' 'c_6' 'c_8' '0.7874' '0.6557314975738525']
    #  ['model_2855' 'c_12' 'm_1' 'c_9' 'c_9' '0.7862' '0.6642679141521454']
    #  ['model_2318' 'c_10' 'm_3' 'c_5' 'c_9' '0.7855' '0.6444030035495758']]

    '''4 cifar10'''
    #c_4,m_2,c_11,c_9
    # total ep: 3600 : 1800/1000/800 exp: 0
    # main_folder = "TMP2_CODE_2/"
    # main_file = "cifar10_model_dict.csv"
    # initial_episode = 1800
    # # Last model is :  ['model_3204', 'm_2', 'c_6', 'c_11', 'c_2', '0.6925', '0.9216696003913879']
    # # Mean of initial:  1800  is -->  0.722534833333334
    # # The list of top  5  are as follows:
    # # [['model_3027' 'c_11' 'm_2' 'c_11' 'c_9' '0.8009' '0.6226186256408691']
    # # ['model_2670' 'c_12' 'm_3' 'c_11' 'c_12' '0.796' '0.6214662475109101']
    # # ['model_2379' 'c_1' 'c_12' 'm_3' 'c_9' '0.7957' '0.6192516955852508']
    # # ['model_3055' 'c_7' 'm_2' 'c_11' 'c_9' '0.7931' '0.6512931025028229']
    # # ['model_2694' 'c_9' 'm_3' 'c_10' 'c_8' '0.7919' '0.63528909907341']]

    '''5 cifar10'''
    # c_12,c_6,c_11,c_11
    # total eps: 3800 : 2000/1000/800 exp:50 periodically
    main_folder = "CIFAR10_5/"
    main_file = "cifar10_model_dict.csv"
    initial_episode = 2000
   #  Last model is :  ['model_3096', 'c_12', 'c_5', 'c_11', 'c_10', '0.7239', '0.8542132089614868']
   #  Mean of initial:  2000  is -->  0.7226369500000003
   #  The list of top  5  are as follows:
   # [['model_2467' 'c_11' 'm_3' 'c_9' 'c_12' '0.8017' '0.65138341755867']
   # ['model_2563' 'c_10' 'm_3' 'c_12' 'c_12' '0.7967' '0.623005072593689']
   # ['model_2965' 'c_12' 'm_1' 'c_11' 'c_11' '0.7966' '0.6526033192634583']
   # ['model_2594' 'c_12' 'm_3' 'c_11' 'c_11' '0.7937' '0.6693723202705383']
   # ['model_3014' 'c_10' 'm_1' 'c_12' 'c_3' '0.7896' '0.654450488114357']]
   # val_acc = 0.7501
   #  val_loss = 0.7637545



    file = file_path + file_type + main_folder + main_file
    data = get_data_from_csv(file)
    data = format_data_without_header(data)
    # sorted_data = sort_data(data,SORTING_INDEX)
    print("\nLast model is : ", data[-1])
    file = file.strip('.csv')
    file += "_sorted.csv"
    # result_file_name = save_topology_in_csv(file,sorted_data)

    index_target = INDEX_ACCURACY
    initial_eps_mean = get_mean_initial_ep(data,initial_episode,index_target)
    print("\nMean of initial: ",initial_episode, " is --> ", initial_eps_mean)

    data_without_initial_ep = remove_data_initial_ep(data,initial_episode)
    sorted_data = sort_data(data_without_initial_ep,SORTING_INDEX)
    num_model = 5
    list_top_model = get_top_model(sorted_data,num_model)
    print("\nThe list of top ",num_model," are as follows:\n ",list_top_model)

if __name__ == "__main__":
    main()
