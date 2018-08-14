import numpy as np
import pandas as pd
import seaborn as sns
from sympy import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerBase
import os
from scipy import stats
sns.set(color_codes=True)

INDEX_ACCURACY = -2

def create_dist_chart(data,dataset,save_path):
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

    plt.title('Distribution of Validation Accuracy of Model found by Random Search on {}'.format(dataset))
    plt.xlabel('Validation Accuracy')
    plt.ylabel('Number of Model')
    # plt.show()
    plt.savefig(save_path)

def main():
    CURRENT_WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
    FOLDER = 'MODEL_WITH_RANDOM_SEARCH'

    DATASET_FOLDER = 'MNIST'
    DATASET_FOLDER = 'CIFAR-10'

    INPUT_FILE = 'original_model.csv'

    PATH_DATASET = os.path.join(CURRENT_WORKING_DIR,FOLDER,DATASET_FOLDER)
    INPUT_FILE_WITH_PATH = os.path.join(PATH_DATASET,INPUT_FILE)

    data = pd.read_csv(INPUT_FILE_WITH_PATH)
    data_accuracy = data.iloc[:,INDEX_ACCURACY].values
    create_dist_chart(data_accuracy,DATASET_FOLDER,PATH_DATASET)

if __name__ == "__main__":
    main()
