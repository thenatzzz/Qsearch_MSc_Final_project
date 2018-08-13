import sys
sys.path.insert(0,'/homes/nj2217/PROJECT/TMP4_CLEAN_CODE')

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
    with open(input_file, 'r') as inp:
        valid_rows = [row for row in csv.reader(inp) if any(field.strip() for field in row)]

    with open(output_file, 'w') as out:
        csv.writer(out).writerows(valid_rows)

def add_zero(file_name):
    name = file_name.strip('episode_')
    if len(name) == 3:
        name = '0'+name
    elif len(name) == 2:
        name = '00'+name
    elif len(name) == 1:
        name = '000'+name
    return 'episode_'+name

def create_heatmap(file_name,input_folder_loc, output_folder_loc):
    # file_name = 'episode_3115.csv'
    print("initial file_name: ",file_name)
    eps_file = pd.read_csv(file_name)
    eps_file = eps_file.pivot("Layer_type","Layer_number","Q-value")

    cmap = sns.cm.rocket_r
    arr = eps_file.values
    vmin, vmax = arr.min(), arr.max()
    # vmin = 0
    # vmax = 1
    # eps_file = (eps_file - eps_file.mean())/eps_file.std()
    ax = sns.heatmap(eps_file, linewidths = 1, vmin=vmin, vmax=vmax,cmap = cmap, annot=True, fmt='.4g')
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 10)
    ax2 = ax.get_figure()

    # SAVING_DIR = "/homes/nj2217/PROJECT/VISUALIZE/HEATMAP_PNG/"
    SAVING_DIR = output_folder_loc
    # file_name = file_name.strip("/homes/nj2217/PROJECT/VISUALIZE/FORMATTED_EPS_FILE/")
    print("input_folder_loc : ", input_folder_loc)
    file_name = file_name.strip(input_folder_loc)
    # print("file_name2 : ", file_name)

    file_name = file_name.strip(".csv")
    # file_name = "e"+file_name

    print("complete file_name: ", file_name)
    file_name = add_zero(file_name)
    print(file_name)
    png_file = SAVING_DIR + file_name+'.png'
    # png_file = SAVING_DIR + file_name
    print("png_file: ", png_file)
    ax2.savefig(png_file,format='png')
    print("save: ", file_name ," heatmap in png successfully!")
    ax2.clf()
    # ax2.cla()
    # plt.show()

def execute(input_folder_loc, output_folder_loc):
    # DIR = "/homes/nj2217/PROJECT/VISUALIZE/FORMATTED_EPS_FILE"
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

def create_vdo(path,vdo_name, framerate):
   # os.system("cd "+path)
#    os.system("cd /vol/gpudata/nj2217/HEATMAP/HEATMAP_PNG")
    os.system("ffmpeg -framerate "+str(framerate)+" -i episode_%04d.png -c:v libx264 -r 30 -pix_fmt yuv420p "+path+"/"+vdo_name)
    print("Create VDO of HEATMAP png_file named: ", vdo_name," successfully!!!")
    
def main():
    input_file = "q_table0-3499.csv"
    output_file = "out_qtable0_3499.csv"
    input_file = "q_table_cifar10_sq_no_exp.csv"
    output_file = "out_q_table_cifar10_sq_no_exp.csv"
  #  remove_space_csv(input_file,output_file)

    DIR = "/homes/nj2217/PROJECT/VISUALIZE/FORMATTED_EPS_FILE"
    SAVING_DIR = "/homes/nj2217/PROJECT/VISUALIZE/HEATMAP_PNG/"

    DIR = "/vol/gpudata/nj2217/HEATMAP/SNS_Q_TABLE"
    SAVING_DIR = "/vol/gpudata/nj2217/HEATMAP/HEATMAP_PNG/"

    data = get_data_from_csv(output_file)
    PATH = "/vol/gpudata/nj2217/HEATMAP/SNS_Q_TABLE/"
  #  format_data(data, PATH)

    input_folder_loc = DIR
    output_folder_loc = SAVING_DIR
    print("Begin execution!")
  #  execute(input_folder_loc, output_folder_loc)
    print("End execution!")

    VDO_NAME = "TEST_HEATMAP.mp4"
    FRAMERATE = 25
    VDO_PATH = "/vol/gpudata/nj2217/HEATMAP/HEATMAP_PNG"
    create_vdo(VDO_PATH,VDO_NAME,FRAMERATE)
    
if "__main__" == __name__:
    main()


# create vdo
# ffmpeg -framerate 25 -i episode_%04d.png -c:v libx264 -r 30 -pix_fmt yuv420p HEATMAP.mp4
