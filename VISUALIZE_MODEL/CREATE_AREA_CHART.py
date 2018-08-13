import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'/homes/nj2217/PROJECT/TMP4_CLEAN_CODE')

from HELPER_FUNCTION import *

ROUND_DIGIT = 6
ACC_INDEX = -2
LOSS_INDEX = -1

def get_column(data,criteria,how_many_model):
    col_list = []
    for index in range(how_many_model):
        if criteria == 'loss':
            col_list.append(float(data[index][LOSS_INDEX]))
        elif criteria == 'accuracy':
            col_list.append(float(data[index][ACC_INDEX]))

    return np.asarray(col_list)

def calculate_avg(index,data,index_criteria):
    avg = 0
    for i in range(index+1):
        avg += float(data[i][index_criteria])
    return avg/(index+1)

def compute_avg_data(data,criteria):
    tmp_list = []
    if criteria == 'accuracy':
        index_criteria = ACC_INDEX
    elif criteria == 'loss':
        index_criteria = LOSS_INDEX

    for index in range(len(data)):
        avg = calculate_avg(index, data,index_criteria)
        tmp_list.append(avg)

    final_list = np.round(tmp_list,ROUND_DIGIT)

    return final_list

def calculate_avg_per_ep(tracking_index,data,index_criteria,episode_interval):
    avg = 0
    for i in range(tracking_index, tracking_index+episode_interval):
        avg += float(data[i][index_criteria])
    return np.round(avg/episode_interval,ROUND_DIGIT)

def compute_episode_data(episode_interval, data, criteria):

    tmp_list = []
    if criteria == 'accuracy':
        index_criteria = ACC_INDEX
    elif criteria == 'loss':
        index_criteria = LOSS_INDEX
    tracking_index = 0
    while tracking_index < len(data):
        avg = calculate_avg_per_ep(tracking_index, data, index_criteria, episode_interval)
        tmp_list.append(avg)
        previous_tracking_index = tracking_index
        tracking_index += episode_interval
        checking_index = tracking_index + episode_interval

        if checking_index > len(data):

            episode_interval = len(data) - tracking_index
            avg = calculate_avg_per_ep(tracking_index, data, index_criteria, episode_interval)
            tmp_list.append(avg)
            break
    return tmp_list

def create_area_chart(data,criteria,list,DATASET = None):

    sns.set_style("whitegrid")

    # Color palette
    blue, = sns.color_palette("muted", 1)


    # Create data
    x = np.arange(len(list))

    # y = get_column(data,criteria,how_many_model)
    y = list

    # Make the plot
    fig, ax = plt.subplots()

    ax.plot(x, y, color=blue, lw=3)
    ax.fill_between(x, 0, y, alpha=.3)

    if DATASET == 'mnist':
        y_ax_range = (0.875,1.0)
    elif DATASET == 'cifar10':
        y_ax_range = (0.69,0.74)
    else:
        y_ax_range = (0,None)
    ax.set(xticklabels=[])
    ax.set(xlim=(0, len(x) - 1), ylim= y_ax_range, xticks=x)

    plt.show()

def format_list_episode(normal_list, episode_list, episode_interval):
    temp_list = []
    count = 0
    count_interval = 0
    while count < len(normal_list):

        previous_count = count
        count += episode_interval
        if count > len(normal_list):
            episode_interval = len(normal_list) - previous_count
        for i in range(episode_interval):
            temp_list.append(episode_list[count_interval])
        count_interval += 1

    return temp_list

def create_area_chart_2_graph(data,criteria,list,list2,x_axis,saved_fig_name,DATASET = None):
    sns.set_style("whitegrid")
    blue, = sns.color_palette("muted", 1)
    # blue, = sns.color_palette("bright", 1)
    # plt.rcParams['axes.facecolor'] = 'white'
    # plt.rcParams['axes.edgecolor'] = 'white'
    # plt.rcParams['grid.alpha'] = 1
    # plt.rcParams['grid.color'] = "#cccccc"
    x = np.arange(len(list))
    y = list

    x2 = np.arange(len(list2))
    y2 = list2
    fig, ax = plt.subplots()
    ax.plot(x, y, color='red', lw=3, label = "Running Average Reward")
    ax.plot(x,y2, color=blue, lw=4, label = 'Average Reward per Epsilon')
    plt.legend(loc='best')

    # ax.fill_between(x, 0, y, alpha=.3)
    ax.fill_between(x, 0, y2, alpha=.3)

    if DATASET == 'mnist':
        y_ax_range = (0.875,1.0)
    elif DATASET == 'cifar10':
        y_ax_range = (0.69,0.74)
    else:
        y_ax_range = (0,None)
    # ax.set(xticklabels=[])
    ax.set(xticklabels=x_axis)

    ax.set(xlim=(0, len(x) - 1), ylim= y_ax_range, xticks=x)
    plt.xlabel('Episode Number')
    plt.ylabel('Accuracy')

    # plt.annotate('', xy = [10,20],xycoords=y2)
    if DATASET == "cifar10":
        dataset = "CIFAR-10"
    elif DATASET == 'mnist':
        dataset = "MNIST"
    else:
        dataset = "Data"
    print(dataset)
    plt.title("Q-Learning Performance on "+dataset)
    # plt.show()
    plt.savefig(saved_fig_name)
    plt.show()

def create_x_axis(data, episode_interval):
    temp_list = []
    num_interval = len(data) // episode_interval

    for i in range(num_interval+1):
        i = i* episode_interval
        temp_list.append(i)

        for j in range(episode_interval-1):

            if len(data) - len(temp_list) < episode_interval and len(temp_list) > len(data):
                for j in range(len(temp_list), len(data)-1):
                    temp_list.append('')
                continue

            temp_list.append('')

    print(len(temp_list))
    return temp_list

def main():
    criteria = 'accuracy'
    '''
    file_name = "fixed_model_dict.csv"
    DATASET = 'mnist'
    '''
    '''
    # file_name = "COMPLETE_CIFAR10.csv"
    # DATASET = 'cifar10'
    '''

    '''
    # MNIST: CLEAN_CODE
    file_path = "/homes/nj2217/PROJECT/VISUALIZE_MODEL/FINISHED_MODEL/MNIST/CLEAN_CODE/"
    file = "fixed_model_dict.csv"
    file = "original_model_dict.csv"
    file_name = file_path + file
    DATASET = 'mnist'
    '''
    '''
    # MNIST: MNIST_GPU
    file_path = "/homes/nj2217/PROJECT/VISUALIZE_MODEL/FINISHED_MODEL/MNIST/MNIST_GPU/"
    file = "episode_fixed_model.csv"
    file_name = file_path + file
    DATASET = 'mnist'
    '''

    # MNIST: MNIST_GPU_UPDATE_EXP
    file_path = "/homes/nj2217/PROJECT/VISUALIZE_MODEL/FINISHED_MODEL/MNIST/MNIST_GPU_UPDATE_EXP/"
    file = "episode_fixed_model.csv"
    file = 'test.csv'
    file_name = file_path + file
    DATASET = 'mnist'

    '''
    # MNIST: MNIST_GPU_UPDATE_EXP_PERIOD
    file_path = "/homes/nj2217/PROJECT/VISUALIZE_MODEL/FINISHED_MODEL/MNIST/MNIST_GPU_UPDATE_EXP_PERIOD/"
    file = "episode_fixed_model.csv"
    # file = 'test.csv'
    file_name = file_path + file
    DATASET = 'mnist'
    '''
    '''
    #CIFAR-10: TMP_CODE
    file_path ="/homes/nj2217/PROJECT/VISUALIZE_MODEL/FINISHED_MODEL/CIFAR-10/TMP_CODE/"
    file = "COMPLETE_CIFAR10.csv"
    file_name = file_path + file
    DATASET = 'cifar10'

    #CIFAR-10: TMP2_CLEAN_CODE
    # lost file
    '''
    '''
    #CIFAR-10: TMP3_CLEAN_CODE
    file_path ="/homes/nj2217/PROJECT/VISUALIZE_MODEL/FINISHED_MODEL/CIFAR-10/TMP3_CLEAN_CODE/"
    file = "COMPLETE_CIFAR10.csv"
    file_name = file_path + file
    DATASET = 'cifar10'
    '''
    '''
    #CIFAR-10: TMP2_CODE_2
    file_path ="/homes/nj2217/PROJECT/VISUALIZE_MODEL/FINISHED_MODEL/CIFAR-10/TMP_CODE_2/"
    file = "COMPLETE_CIFAR10.csv"
    file_name = file_path + file
    DATASET = 'cifar10'
    '''
    '''
    #CIFAR-10: TMP2_CODE_2
    file_path ="/homes/nj2217/PROJECT/VISUALIZE_MODEL/FINISHED_MODEL/CIFAR-10/TMP2_CODE_2/"
    file = "COMPLETE_CIFAR10.csv"
    file = "COMPLETE_CIFAR10_modified.csv"
    file_name = file_path + file
    DATASET = 'cifar10'
    '''

    data = get_data_from_csv(file_name)
    data = format_data_without_header(data)
    list = compute_avg_data(data,criteria)
    # create_area_chart(data,criteria,list,DATASET)

    episode_interval = 300
    # episode_interval = 200


    file_name = file_name.strip('.csv')
    file_name = file_name + '.png'
    saved_fig_name = file_name
    list_2 = compute_episode_data(episode_interval,data,criteria)
    list_2 = format_list_episode(list, list_2, episode_interval)
    # create_area_chart(data,criteria,list_2,DATASET)
    x_axis_eps_interval = create_x_axis(list_2,episode_interval)
    create_area_chart_2_graph(data,criteria,list,list_2,x_axis_eps_interval,saved_fig_name,DATASET)

if __name__ == "__main__":
    main()
