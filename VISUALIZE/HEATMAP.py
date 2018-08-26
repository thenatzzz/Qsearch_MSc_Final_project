import sys
sys.path.insert(0,'/homes/nj2217/FINAL_PROJECT/MAIN')

from HELPER_FUNCTION import *
from FORMAT_TO_SNS_FORMAT import format_data

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os, os.path
import cv2

def remove_space_csv(input_file, output_file):
    ############################################################################
    # FUNCTION DESCRIPTION: remove spaces between each episode of Qtable
    ############################################################################

    with open(input_file, 'r') as inp:
        valid_rows = [row for row in csv.reader(inp) if any(field.strip() for field in row)]

    with open(output_file, 'w') as out:
        csv.writer(out).writerows(valid_rows)

def add_zero(file_name):
    ############################################################################
    # FUNCTION DESCRIPTION: format file name by adding zero in order to get 4 digit format
    ############################################################################

    name = file_name.strip('episode_')
    if len(name) == 3:
        name = '0'+name
    elif len(name) == 2:
        name = '00'+name
    elif len(name) == 1:
        name = '000'+name
    return 'episode_'+name

def get_episode_number(file_path):
    ############################################################################
    # FUNCTION DESCRIPTION: get episode number from file
    ############################################################################

    start_index = file_path.index("episode_")
    episode_name = file_path[start_index:-1]
    return episode_name.strip('.csv')

def create_heatmap(file_name,input_folder_loc, output_folder_loc):
    ############################################################################
    # FUNCTION DESCRIPTION: create single heatmap png from 1 file csv
    ############################################################################
    print("initial file_name: ",file_name)
    eps_file = pd.read_csv(file_name)
    eps_file = eps_file.pivot("Layer_type","Layer_number","Q-value")

    cmap = sns.cm.rocket_r
    arr = eps_file.values
    vmin, vmax = arr.min(), arr.max()
    # vmin = 0   ### uncomment to set heatmap minimum value
    # vmax = 1   ### uncomment to set heatmap maximum value
    # eps_file = (eps_file - eps_file.mean())/eps_file.std()  ### uncommet to implement data normalization
    ax = sns.heatmap(eps_file, linewidths = 1, vmin=vmin, vmax=vmax,cmap = cmap, annot=True, fmt='.4g')
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 10)
    ax2 = ax.get_figure()

    title_name = get_episode_number(file_name)
    ax.set_title(title_name)

    SAVING_DIR = output_folder_loc
    print("input_folder_loc : ", input_folder_loc)
    file_name = file_name.strip(input_folder_loc)

    file_name = file_name.strip(".csv")

    print("complete file_name: ", file_name)
    file_name = add_zero(file_name)
    png_file = SAVING_DIR + file_name+'.png'
    print("png_file: ", png_file)
    ax2.savefig(png_file,format='png')
    print("save: ", file_name ," heatmap in png successfully!")
    ax2.clf()
    # ax2.cla()
    # plt.show()    ### uncomment to show graph

def execute(input_folder_loc, output_folder_loc):
    ############################################################################
    # FUNCTION DESCRIPTION: create bunch of heatmap from csv file in input folder
    #                       and output heatmap png file to output folder
    ############################################################################
    DIR = input_folder_loc
    number_of_file = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    count = 0
    print("number of file" , number_of_file)
    for file in os.listdir(DIR):
        if file.endswith(".csv"):
            formatted_qtable_file = os.path.join(DIR,file)
            print(formatted_qtable_file)
            create_heatmap(formatted_qtable_file,input_folder_loc,output_folder_loc)
            print("Number of file created: ", count)
            print('\n')
            count += 1

def main():
    input_file = "main.csv"
    output_file = "format_main.csv"
    remove_space_csv(input_file,output_file)

    data = get_data_from_csv(output_file)
    OUTPUT_EP_FILE = "/vol/bitbucket/nj2217/FINAL_PROJECT/HEATMAP/EPISODE/"
    format_data(data, OUTPUT_EP_FILE)

    HEATMAP_PNG_DIR = "/vol/bitbucket/nj2217/FINAL_PROJECT/HEATMAP/PNG/"
    HEATMAP_PNG_DIR = "/vol/bitbucket/nj2217/FINAL_PROJECT/HEATMAP/PNG2/"

    input_folder_loc = OUTPUT_EP_FILE
    output_folder_loc = HEATMAP_PNG_DIR
    print("Begin execution!")
    execute(input_folder_loc, output_folder_loc)
    print("End execution!")

if "__main__" == __name__:
    main()

################# TO CREATE VIDEO OF PNG FILES #################################
# REQUIREMENTs: 1. need ffmpeg installed
#               2. execute command in folder that png files are located below to create vdo
# note: can change framerate, file name, vdo output name
# ffmpeg -framerate 25 -i episode_%04d.png -c:v libx264 -r 30 -pix_fmt yuv420p HEATMAP_cifar10.mp4
################################################################################
