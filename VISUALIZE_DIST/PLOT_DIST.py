import numpy as np
import pandas as pd
import seaborn as sns
from sympy import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerBase

from scipy import stats
sns.set(color_codes=True)

INDEX_ACCURACY = -2

def create_dist_chart(data):
    mean = np.mean(data)
    std = np.round(np.std(data),5)
    print("mean {} :: std {}".format(mean,std))

    mean_data = "Mean: {}".format(mean)
    mean_line = plt.axvline(mean, color='red', linestyle='dashed', linewidth=2,label=mean_data)

    median = np.median(data)
    median_data = "Median: {}".format(median)
    median_line = plt.axvline(median, color='green', linestyle='dashed', linewidth=2,label=median_data)

    num_model = "Total number of model: {}".format(len(data))
    num_model_line = mpatches.Patch( label=num_model)

    pm = Symbol(u'±')
    std_data = "Mean {} 1 std.({})".format(pm,std)
    std_line = plt.axvline(mean+std, color= 'violet', linestyle='dashed', linewidth=2, label= std_data)
    std_line = plt.axvline(mean-std, color= 'violet', linestyle='dashed', linewidth=2, label= std_data)

    plt.legend(handles= [mean_line,median_line,std_line,num_model_line])
    sns.distplot(data)

    plt.title('Distribution of Validation Accuracy of Model found by Random Search on CIFAR-10')
    plt.xlabel('Validation Accuracy')
    plt.ylabel('Number of Model')
    plt.show()

def main():
    CURRENT_WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
    FOLDER = 'FINISHED_MODEL'

    DATASET = 'mnist'
    DATASET_FOLDER = 'MNIST'
    # MAIN_FOLDER = 'MNIST_1'
    # MAIN_FOLDER = 'MNIST_2'
    # MAIN_FOLDER = 'MNIST_3'
    # MAIN_FOLDER = 'MNIST_4'

    DATASET = 'cifar10'
    DATASET_FOLDER = 'CIFAR-10'
    # MAIN_FOLDER = 'CIFAR-10_1'
    # MAIN_FOLDER = 'CIFAR-10_2'
    # MAIN_FOLDER = 'CIFAR-10_3'
    # MAIN_FOLDER = 'CIFAR-10_4'
    MAIN_FOLDER = 'CIFAR-10_5'

    MODEL_FOLDER = 'MODEL_DICT'
    INPUT_FILE = 'original_model.csv'

    PATH_DATASET = os.path.join(CURRENT_WORKING_DIR,FOLDER,DATASET_FOLDER,\
                                MAIN_FOLDER)
    MODEL_DICT_FOLDER = 'MODEL_DICT'
    INPUT_FILE_WITH_PATH = os.path.join(PATH_DATASET,MODEL_DICT_FOLDER,INPUT_FILE)

                                
    data = pd.read_csv('original_model.csv')
    data_accuracy = data.iloc[:,INDEX_ACCURACY].values
    create_dist_chart(data_accuracy)

if __name__ == "__main__":
    main()
