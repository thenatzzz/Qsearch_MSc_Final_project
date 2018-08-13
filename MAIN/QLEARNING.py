from TRAIN_MODEL_MNIST import train_model_mnist
from HELPER_FUNCTION import get_data_from_csv, format_data_without_header,\
                            save_list_csv_rowbyrow
from TRAIN_MODEL_CIFAR10 import train_model_cifar10

import numpy as np
import random
import csv
import pandas
import os
import time
from enum import Enum

DATASET = ""

FILE = ""
Q_TABLE_FILE= ""
EPISODE_FILE = ""

MODE = "RANDOMIZED_update"
# MODE = "SEQUENTIAL_update"

UPDATE_FROM_MEM_REPLAY = True
#UPDATE_FROM_MEM_REPLAY = False


LAYER_BIAS_ADJUSTMENT_RATE = 0.2

cont_episode = 2710
MAX_ACTION = 16
MAX_STATE = 4
MAX_NUM_LAYER = 4

INDEX_ACCURACY = -2
INDEX_LOSS = -1
INDEX_MODEL = 0
LAYER_SOFTMAX = 's'

NUM_DIGIT_ROUND = 6
NUM_MODEL_FROM_EXPERIENCE_REPLAY = 50

class Action(Enum):
    c_1 = 0
    c_2 = 1
    c_3 = 2
    c_4 = 3
    c_5 = 4
    c_6 = 5
    c_7 = 6
    c_8 = 7
    c_9 = 8
    c_10 = 9
    c_11 = 10
    c_12 = 11
    m_1 = 12
    m_2 = 13
    m_3 = 14
    s = 15

class State:
    layer_1 = 0
    layer_2 = 1
    layer_3 = 2
    layer_4 = 3

''' Create MODEL corresponding to specific epsilon '''
NUM_LIST = 3
NUM_LIST_1 = 1
NUM_LIST_2 = 2
NUM_LIST_3 = 3
''' Change this to specify number of model corresponding to each specific epsilon'''
NUM_MODEL_1 = 2000
NUM_MODEL_2 = 1000
NUM_MODEL_3 = 800

MAX_EPISODE = NUM_MODEL_1 + NUM_MODEL_2 + NUM_MODEL_3

''' Discount Rate is set to 1.0 as to NOT prioritize any specific layer'''
GAMMA = 1.0

''' Alpha = learning rate '''
MIN_ALPHA = 0.01
ALPHA_LIST = np.linspace(1.0, MIN_ALPHA, MAX_EPISODE) # set alpha at decreasing order
ALPHA_LIST = [MIN_ALPHA]*MAX_EPISODE # set alpha equal to MIN_ALPHA

EPSILON_DICT = {}
EPSILON_LIST_1 = 1
EPSILON_LIST_2 = np.linspace(0.9,0.7,NUM_MODEL_2)
EPSILON_LIST_3 = np.linspace(0.6,0.1, NUM_MODEL_3)
EPSILON_DICT[NUM_LIST_1] = EPSILON_LIST_1
EPSILON_DICT[NUM_LIST_2] = EPSILON_LIST_2
EPSILON_DICT[NUM_LIST_3] = EPSILON_LIST_3

INTERVAL = 100

overall_space = (MAX_STATE, MAX_ACTION)
Q_TABLE = np.zeros(overall_space)
#episode_2709
#c_1,c_2,c_3,c_4,c_5,c_6,c_7,c_8,c_9,c_10,c_11,c_12,m_1,m_2,m_3,s
Q_TABLE = [[0.185325,0.159954,0.191658,0.167693,0.208087,0.175016,0.198239,0.151909,0.182845,0.17471,0.185027,0.317695,0.120346,0.140026,0.111713,0.047107],
[0.315484,0.317231,0.286442,0.275471,0.295349,0.462981,0.292177,0.281923,0.314483,0.277737,0.289698,0.294643,0.246778,0.253373,0.29063,0.065597],
[0.432165,0.433477,0.417343,0.452142,0.416055,0.419662,0.441318,0.482537,0.416111,0.417235,0.615036,0.424215,0.410638,0.392321,0.430969,0.1188],
[0.573483,0.602213,0.560425,0.562545,0.567915,0.584909,0.608368,0.588902,0.627308,0.5645,0.695613,0.598352,0.559972,0.596935,0.601257,0.122556]]

Q_TABLE = np.asarray(Q_TABLE)

def get_file(dataset):
    global FILE
    global Q_TABLE_FILE
    global EPISODE_FILE

    if dataset == 'cifar10':
        FILE = "COMPLETE_CIFAR10.csv"
        Q_TABLE_FILE= "q_table_cifar10.csv"
        EPISODE_FILE = 'episode_cifar10.csv'
    elif dataset == 'mnist':
        FILE = "fixed_model_dict.csv"
        Q_TABLE_FILE= "q_table_mnist.csv"
        EPISODE_FILE = 'episode_mnist.csv'
    else:
        FILE = 'file.csv'
        Q_TABLE_FILE = "q_table.csv"
        EPISODE_FILE = 'episode.csv'

def check_equal(some_list):
    for i in range(len(some_list)):
        if some_list[i] != some_list[0]:
            return False
    return True

def match_epsilon(epsilon):
    if epsilon+1 <= NUM_MODEL_1:
        return EPSILON_DICT[NUM_LIST_1]
    elif epsilon+1 > NUM_MODEL_1 and epsilon+1 <= NUM_MODEL_1+NUM_MODEL_2:
        return EPSILON_DICT[NUM_LIST_2][epsilon-NUM_MODEL_1]
    else:
        return EPSILON_DICT[NUM_LIST_3][epsilon-NUM_MODEL_1-NUM_MODEL_2]

def choose_action(num_layer, epsilon):
    eps = match_epsilon(epsilon)
    if random.uniform(0,1) < eps:
        random_key, random_value = random.choice(list(enumerate(Q_TABLE[num_layer])))
        return random_key,random_value
    else:
        if check_equal(Q_TABLE[num_layer]):
            max_key, max_value = random.choice(list(enumerate(Q_TABLE[num_layer])))
        else:
            max_key = Q_TABLE[num_layer].argmax()
            max_value = Q_TABLE[num_layer][max_key]
        return max_key,max_value

def choose_action_exp(data,num_layer,episode,tracking_index):
    if episode < NUM_MODEL_1:
        if num_layer ==0:
            if MODE == "RANDOMIZED_update":
                tracking_index = random.randint(0,len(data)-1)
            elif MODE == "SEQUENTIAL_update":
                tracking_index = episode
            else:
                tracking_index = random.randint(0,len(data)-1)
        action = str(data[tracking_index][num_layer+1])

        for enum_action in Action:
            tracking_action = 'None'
            if action == enum_action.name:
                tracking_action = enum_action.value
                break

        if tracking_action == 'None':
            return
        value = Q_TABLE[num_layer][tracking_action]
        return tracking_action, value, tracking_index
    else:
        tracking_index = 0
        action, value = choose_action(num_layer, episode)
        return action,value,tracking_index

def fn_format_action_array(action_array):
    new_action_array = action_array[:]
    length_action_array = len(action_array)
    if length_action_array != MAX_NUM_LAYER:
        for i in range(MAX_NUM_LAYER-length_action_array):
            new_action_array.append('-')
    return new_action_array

def fix_layer_acc_bias(max_value_next_action,num_layer,action_array_1):
    if action_array_1[num_layer] == LAYER_SOFTMAX:
        max_value_next_action *= LAYER_BIAS_ADJUSTMENT_RATE
    return max_value_next_action

def update_qtable_from_mem_replay(data,num_model,dataset):
    global MODE
    MODE = "RANDOMIZED_update"
    for i in range(num_model):
        index = 0
        num_layer = 0
        action_array = []
        action_array_1 = []
        action = 0
        num_layer = 0
        alpha = ALPHA_LIST[i]
        # print(i)
        while Action(action).name != LAYER_SOFTMAX and num_layer < MAX_NUM_LAYER:
            action,value_action, index = choose_action_exp(data,num_layer,i,index)
            action_array.append(action)
            action_array_1 = translate_action_array(action_array)
            # print("action_array_1: ",action_array_1)
            max_value_next_action = get_next_value(data,num_layer, action_array_1,dataset)
            max_value_next_action = fix_layer_acc_bias(max_value_next_action,num_layer,action_array_1)

            Q_TABLE[num_layer][action] = Q_TABLE[num_layer][action] + \
                        alpha*(GAMMA*max_value_next_action-Q_TABLE[num_layer][action])
            Q_TABLE[num_layer] = round_value(Q_TABLE[num_layer])
            num_layer += 1


def train_new_model(data,action_array,dataset):
    #print("______________________________________________________")
    #print("_________________ CANNOT FIND A MATCH ________________")
    #print("______________________________________________________")
    # num_model = NUM_MODEL_FROM_EXPERIENCE_REPLAY
    # if UPDATE_FROM_MEM_REPLAY:
    #     update_qtable_from_mem_replay(data,num_model,dataset)
    # return 1

    if dataset == "cifar10":
        return train_model_cifar10(action_array)
    elif dataset == "mnist":
        return train_model_mnist(action_array)

def get_accuracy(data,action_array,dataset):
    new_action_array = action_array[:]
    temp_action_array =  []
    temp_action_array = fn_format_action_array(new_action_array)
    for index in range(len(data)):

        if np.array_equal(temp_action_array, data[index][1:INDEX_ACCURACY]):
        #    print('+++++++++++++++++++++++++++++++++++++++++++++++')
        #    print("+++++++++ THERE IS A MATCH !! +++++++++++++++++")
        #    print("at model number: ", data[index][0])
        #    print('+++++++++++++++++++++++++++++++++++++++++++++++')
        #    print("Accuray of model: ",data[index][INDEX_ACCURACY])
            return float(data[index][INDEX_ACCURACY])
    return train_new_model(data,temp_action_array,dataset)

def get_next_value(data,num_layer,action_array,dataset):
    '''set num_layer to next layer '''
    num_layer += 1
    if num_layer == MAX_NUM_LAYER or action_array[-1] == LAYER_SOFTMAX:
        return get_accuracy(data,action_array,dataset)
    else:
        max_key = Q_TABLE[num_layer].argmax()
        max_value = Q_TABLE[num_layer][max_key]
        return max_value

def translate_action_array(action_array):
    temp_array = []
    for i in range(len(action_array)):
        temp_array.append(Action(action_array[i]).name)
    return temp_array

def round_value(temp_q_table):
    return np.round(temp_q_table,NUM_DIGIT_ROUND)

def get_best_action(table):
    dict_1 = {}
    tup_1 = ()
    for i in range(MAX_STATE):
        max_key = table[i].argmax()
        max_value = table[i][max_key]
        max_value =  round(max_value,NUM_DIGIT_ROUND)
        tup_1 = (Action(max_key).name, max_value)
        key = "Layer "+ str(i+1)
        dict_1[key] = tup_1
    return dict_1

def get_avg_accuray(table):
    dict_1 = {}
    for i in range(MAX_STATE):
        key = "Layer "+ str(i+1)
        avg_value = sum(table[i])/len(table[i])
        avg_value = round(avg_value, NUM_DIGIT_ROUND)
        dict_1[key] = avg_value
    return dict_1

def save_q_table(episode,Q_TABLE,dataset):
    if len(Q_TABLE_FILE) > 0 :
        file_name = Q_TABLE_FILE
    elif dataset == "cifar10":
        file_name = "q_table_cifar10.csv"
    elif dataset == "mnist":
        file_name = "q_table_mnist.csv"

    list_of_data = Q_TABLE[:]
    episode_name = 'episode_'+ str(episode)
    csv_columns = [[],[episode_name],['c_1','c_2','c_3','c_4','c_5','c_6','c_7','c_8','c_9','c_10','c_11','c_12','m_1','m_2','m_3','s']]
    data_list = csv_columns + list(Q_TABLE)

    myFile = open(file_name,'a')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(data_list)

def update_data(data,dataset):
    tmp_data = ""
    tmp_data = get_data_from_csv(FILE)
    tmp_data = format_data_without_header(tmp_data)
    data = tmp_data[:]
    return data

def save_finished_episode(episode_number,data,action_array):
    temp_array = []
    temp_action_array = ""
    temp_action_array = fn_format_action_array(action_array)
    # print("inside save_fin: ",episode_number, " : ",temp_action_array)

    if episode_number == 0:
        temp_array.append(['EPISODE_NUMBER','LAYER_1','LAYER_2','LAYER_3','LAYER_4','ACCURACY','LOSS','MODEL_MATCHED'])
    for datum in data:
        action_datum = datum[1:-2]
        temp_datum = []
        if temp_action_array == action_datum:
            temp_datum = ['episode_'+str(episode_number)] + action_datum + \
                        [datum[INDEX_ACCURACY]]+[datum[INDEX_LOSS]]+[datum[INDEX_MODEL]]
            temp_array.append(temp_datum)

    save_list_csv_rowbyrow(EPISODE_FILE, temp_array,'a')

def create_exp_replay_interval():
    temp_list = []
    num_interval = MAX_EPISODE// INTERVAL
    print("num_interval: ", num_interval)
    for i in range(num_interval):
        temp_list.append(INTERVAL*i + NUM_MODEL_1)
    return temp_list

def match_exp_replay_interval(episode_number,exp_replay_interval):
    for i in range(len(exp_replay_interval)):
        if episode_number == exp_replay_interval[i]:
            return True

    return False

def run_q_learning(data,dataset):

    get_file(dataset)
    exp_replay_interval = create_exp_replay_interval()
    print(exp_replay_interval)
    for episode_number in range(cont_episode,MAX_EPISODE):
        action = 0
        num_layer = 0
        alpha = ALPHA_LIST[episode_number]
        action_array = []
        action_array_1 = []
        index = 0
        # print('alpha: ',alpha)
        while Action(action).name != LAYER_SOFTMAX and num_layer < MAX_NUM_LAYER:
            # action,value_action = choose_action(num_layer, i)
            action,value_action, index = choose_action_exp(data,num_layer,episode_number,index)
            action_array.append(action)
            action_array_1 = translate_action_array(action_array)
        #    print("action_array_1: ",action_array_1)
            # print(i)
            max_value_next_action = get_next_value(data,num_layer, action_array_1,dataset)
            max_value_next_action = fix_layer_acc_bias(max_value_next_action,num_layer,action_array_1)

            Q_TABLE[num_layer][action] = Q_TABLE[num_layer][action] + \
                        alpha*(GAMMA*max_value_next_action-Q_TABLE[num_layer][action])
            Q_TABLE[num_layer] = round_value(Q_TABLE[num_layer])
            num_layer += 1
        # print("ZZZZZZZ index ZZZZZZZZZZZZZ: ", index)
        print(action_array_1)

        if UPDATE_FROM_MEM_REPLAY and match_exp_replay_interval(episode_number,exp_replay_interval):
            num_model = NUM_MODEL_FROM_EXPERIENCE_REPLAY
            # print("UPDATEEEEEEEEEEEEEEEEEEEEEE at episode number:", episode_number )
            update_qtable_from_mem_replay(data,num_model,dataset)

        print("$$$$$$$$$$$$$$ EPISODE: ", episode_number, " $$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        save_q_table(episode_number,Q_TABLE,dataset)
        data = update_data(data,dataset)
        save_finished_episode(episode_number,data,action_array_1)
        # print(Q_TABLE)
        print('\n')
        # time.sleep(1)
    print("Best accuracy: ",get_best_action(Q_TABLE))
    print("Avg accuracy: ",get_avg_accuray(Q_TABLE))
    return get_best_action(Q_TABLE)
