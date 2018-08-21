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

    sns.distplot(data[:2000], label='epsilon 1 (Random Search):\n 2000 episodes',color='dodgerblue')
    # sns.distplot(data[2000:3000], label='epsilon 0.9-0.7:\n 1000 episodes',color='darkorange')
    # sns.distplot(data[3000:-1], label='epsilon 0.6-0.1:\n 800 episodes',color='mediumseagreen')

    sns.distplot(data[3750:3800], label='epsilon 0.1:\n 150 episodes',color='mediumseagreen')

    plt.legend()
    plt.title('Model Accuracy Distribution on {} dataset'.format(dataset))
    plt.xlabel('Validation Accuracy')
    plt.ylabel('Number of Model')
    # plt.show()
    plt.savefig(save_path)

def main():
    CURRENT_WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
    FOLDER = 'FINISHED_MODEL'

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
