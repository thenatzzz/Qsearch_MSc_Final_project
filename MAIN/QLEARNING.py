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

''' Specify dataset to use '''
DATASET = ""

''' Specify file name: main file, q_table file and episode file '''
FILE = ""
Q_TABLE_FILE= ""
EPISODE_FILE = ""

''' Experience Replay update mode: 1 update Q-table after the agent finishes training new model.'''
UPDATE_AFTER_FIND_NEW_MODEL = False

NUM_MODEL_AFTER_FIND_NEW_MODEL = 5
MODE = "RANDOMIZED_update"
# MODE = "SEQUENTIAL_update"

''' Experience Replay update mode: 2 update Q-table in periodic manner (intervals)'''
UPDATE_FROM_MEM_REPLAY_PERIODIC = True

NUM_MODEL_PERIODIC = 50
INTERVAL = 100

''' Layer bias adjustment for Softmax Layer '''
LAYER_BIAS_ADJUSTMENT_RATE = 0.2

cont_episode = 0

MAX_ACTION = 16
MAX_STATE = 4
MAX_NUM_LAYER = 4

INDEX_ACCURACY = -2
INDEX_LOSS = -1
INDEX_MODEL = 0
LAYER_SOFTMAX = 's'

NUM_DIGIT_ROUND = 6

POSSIBLE_ACTIONS = ['c_1','c_2','c_3','c_4','c_5','c_6','c_7',\
                'c_8','c_9','c_10','c_11','c_12','m_1','m_2','m_3','s']
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


overall_space = (MAX_STATE, MAX_ACTION)
Q_TABLE = np.zeros(overall_space)

def get_file(dataset):
    ############################################################################
    # FUNCTION DESCRIPTION: get csv file according to specified dataset
    ############################################################################
    global FILE
    global Q_TABLE_FILE
    global EPISODE_FILE

    if dataset == 'cifar10':
        FILE = "cifar10_model.csv"
        Q_TABLE_FILE= "q_table_cifar10.csv"
        EPISODE_FILE = 'episode_cifar10.csv'
    elif dataset == 'mnist':
        FILE = "mnist_model.csv"
        Q_TABLE_FILE= "q_table_mnist.csv"
        EPISODE_FILE = 'episode_mnist.csv'
    else:
        FILE = 'file.csv'
        Q_TABLE_FILE = "q_table.csv"
        EPISODE_FILE = 'episode.csv'

def check_equal(some_list):
    ############################################################################
    # FUNCTION DESCRIPTION: check if the element is same or not
    ############################################################################
    for i in range(len(some_list)):
        if some_list[i] != some_list[0]:
            return False
    return True

def match_epsilon(epsilon):
    ############################################################################
    # FUNCTION DESCRIPTION: function to match epsilon to specified episode
    ############################################################################
    if epsilon+1 <= NUM_MODEL_1:
        return EPSILON_DICT[NUM_LIST_1]
    elif epsilon+1 > NUM_MODEL_1 and epsilon+1 <= NUM_MODEL_1+NUM_MODEL_2:
        return EPSILON_DICT[NUM_LIST_2][epsilon-NUM_MODEL_1]
    else:
        return EPSILON_DICT[NUM_LIST_3][epsilon-NUM_MODEL_1-NUM_MODEL_2]

def choose_action(num_layer, epsilon):
    ############################################################################
    # FUNCTION DESCRIPTION: function to choose action with epsilon greedy strategy
    #                       or random strategy
    ############################################################################
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
    ############################################################################
    # FUNCTION DESCRIPTION: function to choose action from models sampled from
    #                       experience replay to update Q-table
    ############################################################################
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
    ############################################################################
    # FUNCTION DESCRIPTION: function to format/ complete topology array
    ############################################################################
    new_action_array = action_array[:]
    length_action_array = len(action_array)
    if length_action_array != MAX_NUM_LAYER:
        for i in range(MAX_NUM_LAYER-length_action_array):
            new_action_array.append('-')
    return new_action_array

def fix_layer_acc_bias(max_value_next_action,num_layer,action_array_1):
    ############################################################################
    # FUNCTION DESCRIPTION: function to fix layer bias (as softmax layer tends
    #                       to habe high value than others)
    ############################################################################
    if action_array_1[num_layer] == LAYER_SOFTMAX:
        max_value_next_action *= LAYER_BIAS_ADJUSTMENT_RATE
    return max_value_next_action

def update_qtable_from_mem_replay(data,num_model,dataset):
    ############################################################################
    # FUNCTION DESCRIPTION: function to update Q-table by sampling models from
    #                       experience replay
    ############################################################################
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

        while Action(action).name != LAYER_SOFTMAX and num_layer < MAX_NUM_LAYER:
            action,value_action, index = choose_action_exp(data,num_layer,i,index)
            action_array.append(action)
            action_array_1 = translate_action_array(action_array)
            max_value_next_action = get_next_value(data,num_layer, action_array_1,dataset)
            max_value_next_action = fix_layer_acc_bias(max_value_next_action,num_layer,action_array_1)

            Q_TABLE[num_layer][action] = Q_TABLE[num_layer][action] + \
                        alpha*(GAMMA*max_value_next_action-Q_TABLE[num_layer][action])
            Q_TABLE[num_layer] = round_value(Q_TABLE[num_layer])
            num_layer += 1


def train_new_model(data,action_array,dataset):
    ############################################################################
    # FUNCTION DESCRIPTION: function to train new model if Q agent unable to find
    #                       from the experience replay.
    ############################################################################

    if UPDATE_AFTER_FIND_NEW_MODEL:
        update_qtable_from_mem_replay(data,NUM_MODEL_AFTER_FIND_NEW_MODEL,dataset)

    if dataset == "cifar10":
        return train_model_cifar10(action_array)
    elif dataset == "mnist":
        return train_model_mnist(action_array)

def get_accuracy(data,action_array,dataset):
    ############################################################################
    # FUNCTION DESCRIPTION: function to get accuracy if there are already trained model
    #                       in experience replay else need to train new model
    #                       and get new accuracy
    ############################################################################
    new_action_array = action_array[:]
    temp_action_array =  []
    temp_action_array = fn_format_action_array(new_action_array)
    for index in range(len(data)):

        if np.array_equal(temp_action_array, data[index][1:INDEX_ACCURACY]):
            return float(data[index][INDEX_ACCURACY])

    return train_new_model(data,temp_action_array,dataset)

def get_next_value(data,num_layer,action_array,dataset):
    ############################################################################
    # FUNCTION DESCRIPTION: function to get new action value according to Bellman
    #                       update equation
    ############################################################################
    '''set num_layer to next layer '''
    num_layer += 1
    if num_layer == MAX_NUM_LAYER or action_array[-1] == LAYER_SOFTMAX:
        return get_accuracy(data,action_array,dataset)
    else:
        max_key = Q_TABLE[num_layer].argmax()
        max_value = Q_TABLE[num_layer][max_key]
        return max_value

def translate_action_array(action_array):
    ############################################################################
    # FUNCTION DESCRIPTION: function to format action array
    ############################################################################
    temp_array = []
    for i in range(len(action_array)):
        temp_array.append(Action(action_array[i]).name)
    return temp_array

def round_value(temp_q_table):
    ############################################################################
    # FUNCTION DESCRIPTION: function to round value according to specified digit
    ############################################################################
    return np.round(temp_q_table,NUM_DIGIT_ROUND)

def get_best_action(table):
    ############################################################################
    # FUNCTION DESCRIPTION: function to choose the function with highest value
    ############################################################################
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
    ############################################################################
    # FUNCTION DESCRIPTION: function to get average accuracy
    ############################################################################
    dict_1 = {}
    for i in range(MAX_STATE):
        key = "Layer "+ str(i+1)
        avg_value = sum(table[i])/len(table[i])
        avg_value = round(avg_value, NUM_DIGIT_ROUND)
        dict_1[key] = avg_value
    return dict_1

def save_q_table(episode,Q_TABLE,dataset):
    ############################################################################
    # FUNCTION DESCRIPTION: function to save Q-table in csv file
    ############################################################################
    if len(Q_TABLE_FILE) > 0 :
        file_name = Q_TABLE_FILE
    elif dataset == "cifar10":
        file_name = "q_table_cifar10.csv"
    elif dataset == "mnist":
        file_name = "q_table_mnist.csv"

    list_of_data = Q_TABLE[:]
    episode_name = 'episode_'+ str(episode)
    csv_columns = [[],[episode_name],POSSIBLE_ACTIONS]
    data_list = csv_columns + list(Q_TABLE)

    myFile = open(file_name,'a')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(data_list)

def update_data(data,dataset):
    ############################################################################
    # FUNCTION DESCRIPTION: function to update experience replay into on-going processing
    #                       data after the agent puts new model into experience replay
    ############################################################################
    tmp_data = ""
    tmp_data = get_data_from_csv(FILE)
    tmp_data = format_data_without_header(tmp_data)
    data = tmp_data[:]
    return data

def save_finished_episode(episode_number,data,action_array):
    ############################################################################
    # FUNCTION DESCRIPTION: function to save finished episode into csv file
    ############################################################################
    temp_array = []
    temp_action_array = ""
    temp_action_array = fn_format_action_array(action_array)

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
    ############################################################################
    # FUNCTION DESCRIPTION: function to create interval for experience replay
    ############################################################################
    temp_list = []
    num_interval = MAX_EPISODE// INTERVAL
    print("num_interval: ", num_interval)
    for i in range(num_interval):
        temp_list.append(INTERVAL*i + NUM_MODEL_1)
    return temp_list

def match_exp_replay_interval(episode_number,exp_replay_interval):
    ############################################################################
    # FUNCTION DESCRIPTION: function to match experience replay to specified interval
    ############################################################################
    for i in range(len(exp_replay_interval)):
        if episode_number == exp_replay_interval[i]:
            return True
    return False

def run_q_learning(data,dataset):
    ############################################################################
    # FUNCTION DESCRIPTION: main function to run Q-Search
    ############################################################################
    get_file(dataset)
    exp_replay_interval = create_exp_replay_interval()

    for episode_number in range(cont_episode,MAX_EPISODE):
        action = 0
        num_layer = 0
        alpha = ALPHA_LIST[episode_number]
        action_array = []
        action_array_1 = []
        index = 0

        while Action(action).name != LAYER_SOFTMAX and num_layer < MAX_NUM_LAYER:
            action,value_action, index = choose_action_exp(data,num_layer,episode_number,index)
            action_array.append(action)
            action_array_1 = translate_action_array(action_array)
            max_value_next_action = get_next_value(data,num_layer, action_array_1,dataset)
            max_value_next_action = fix_layer_acc_bias(max_value_next_action,num_layer,action_array_1)

            Q_TABLE[num_layer][action] = Q_TABLE[num_layer][action] + \
                        alpha*(GAMMA*max_value_next_action-Q_TABLE[num_layer][action])
            Q_TABLE[num_layer] = round_value(Q_TABLE[num_layer])
            num_layer += 1

        if UPDATE_FROM_MEM_REPLAY_PERIODIC and match_exp_replay_interval(episode_number,exp_replay_interval):

            update_qtable_from_mem_replay(data,NUM_MODEL_PERIODIC,dataset)
        print(Q_TABLE)
        print("$$$$$$$$$$$$$$ EPISODE: ", episode_number, " $$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        save_q_table(episode_number,Q_TABLE,dataset)
        data = update_data(data,dataset)
        save_finished_episode(episode_number,data,action_array_1)
        print('\n')

    print("Best accuracy: ",get_best_action(Q_TABLE))
    print("Avg accuracy: ",get_avg_accuray(Q_TABLE))
    return get_best_action(Q_TABLE)
