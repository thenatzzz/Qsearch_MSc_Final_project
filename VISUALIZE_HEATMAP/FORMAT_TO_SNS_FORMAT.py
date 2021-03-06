import sys
sys.path.insert(0,'/homes/nj2217/FINAL_PROJECT/MAIN')

from HELPER_FUNCTION import *
import numpy as np

INDEX_MODEL = 0
INDEX_ACTION = 1
INDEX_LAYER_1 = 2
INDEX_LAYER_2 = 3
INDEX_LAYER_3 = 4
INDEX_LAYER_4 = 5
QTABLE_LENGTH = 6

def begin_with_episode_word(first_element):
    ###########################################################################
    # FUNCTION DESCRIPTION: return true if the input word is 'episode'
    ############################################################################
    if first_element[0:7] == 'episode':
        return True
    else:
        return False

def check_begining_ending(single_data):
    ############################################################################
    # FUNCTION DESCRIPTION: check the beginning of each episode in Q-table
    ############################################################################
    if begin_with_episode_word(single_data[0]):
        return True
    elif len(single_data) == 1:
        return True
    else:
        return False

def format_action_name(action):
    action = action.replace('_','')
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    action = action.capitalize()
    action = action.translate(SUB)
    return action

def format_one_layer(index,value_in_one_layer, layer_action):
    ############################################################################
    # FUNCTION DESCRIPTION: format each layer of qtable per episode into this
    #                       format: [ layer, action, qtable_val]
    ############################################################################
    layer_num = index -1
    temp_list = []
    final_list = []
    for i in range(len(value_in_one_layer)):
        action = layer_action[i]
        action = format_action_name(action)
        temp_list = [action,layer_num,value_in_one_layer[i] ]
        # print(temp_list)
        final_list.append(temp_list)
    return final_list

def format_data_per_eps(single_qtable):
    ############################################################################
    # FUNCTION DESCRIPTION: main function to format Qtable for each episode
    ############################################################################
    eps_name = single_qtable[INDEX_MODEL][0]

    format_array = []
    for index in range(INDEX_LAYER_1, INDEX_LAYER_4+1):
        format_array += format_one_layer(index,single_qtable[index][:-1], single_qtable[INDEX_ACTION])

    return eps_name,format_array

def is_complete_single_qtable(single_qtable):
    ############################################################################
    # FUNCTION DESCRIPTION: return true if single_qtable is complete
    ############################################################################
    if len(single_qtable) == QTABLE_LENGTH:
        return True
    else:
        return False

def add_index_col(formatted_list):
    for i in range(len(formatted_list)):
        formatted_list[i] = [str(i)]+formatted_list[i]
    return formatted_list

def save_single_eps_in_file(file_name, single_qtable,PATH):
    data = [["Layer Type","Layer Number","Q-value" ]] + single_qtable
    data = np.asarray(data)
    path = PATH
    output_file = path+file_name+".csv"
    output_file = save_list_csv_rowbyrow(output_file,data,'w')

def format_data(data, PATH):
    ############################################################################
    # FUNCTION DESCRIPTION: format each episode of Q-table in Seaborn data format
    #                       into individual file and save into PATH
    ############################################################################
    count = 0
    temp_array = []
    final_list = []
    num_eps_saved = 0
    for i in range(len(data)):
        if check_begining_ending(data[i]):
            count += 1
            temp_array = []

        if count == 2:
            count = 0
            temp_array = []
            temp_array.append(data[i])
            continue

        temp_array.append(data[i])
        if is_complete_single_qtable(temp_array):
            eps_name,final_list = format_data_per_eps(temp_array)
            save_single_eps_in_file(eps_name,final_list,PATH)
            print("save ----> ", eps_name, " file successfully!")
            num_eps_saved += 1

    print("Num of episode saved : ",num_eps_saved)

def main():
    # file = "out_qtable0_3499.csv"
    # file = "testqtable.csv"
    file = "q_table_cifar10.csv"
    data = get_data_from_csv(file)

    # PATH = "/homes/nj2217/PROJECT/VISUALIZE/FORMATTED_EPS_FILE/"
    # PATH = "/vol/gpudata/nj2217/HEATMAP/SNS_Q_TABLE/"

    CURRENT_WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
    # INPUT_FILE_WITH_PATH = os.path.join(PATH_DATASET,INPUT_FILE)
    OUTPUT_PATH= CURRENT_WORKING_DIR
    format_data(data, OUTPUT_PATH)

if __name__ == "__main__":
    main()
