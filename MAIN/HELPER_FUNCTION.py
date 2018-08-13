import csv
import os

KEY_MODEL = 'Model'
KEY_FIRST_LAYER = '1st Layer'
KEY_SECOND_LAYER = '2nd Layer'
KEY_THIRD_LAYER = '3rd Layer'
KEY_FORTH_LAYER = '4th Layer'
KEY_ACCURACY = 'Accuracy'
KEY_LOSS = 'Loss'

LAYER_SOFTMAX = 's'

MODEL_TAG_HEADER= [KEY_MODEL,KEY_FIRST_LAYER,KEY_SECOND_LAYER,KEY_THIRD_LAYER,KEY_FORTH_LAYER,KEY_ACCURACY,KEY_LOSS]
MAX_LENGTH_HEADER = 7

INDEX_MODEL = 0
INDEX_FIRST_LAYER = 1
INDEX_SECOND_LAYER = 2
INDEX_THIRD_LAYER = 3
INDEX_FORTH_LAYER = 4
INDEX_ACCURACY = -2
INDEX_LOSS = -1
INDEX_LAST = -1

def get_data_from_csv(file_name):
    list_data = []
    with open(file_name, 'rt',encoding='utf8') as f:
        reader = csv.reader(f)
        for row in reader:
            list_data.append(row)
    return list_data[:]

def format_data_without_header(data):
    return data[INDEX_FIRST_LAYER:]

def get_topology_only(single_model):
    return single_model[INDEX_FIRST_LAYER:INDEX_ACCURACY]

def check_complete_model(single_model):
    if len(single_model) == MAX_LENGTH_HEADER:
        return True
    else:
        return False

def count_model_layer(model_from_csv):
    count = 0
    for i in range(len(model_from_csv)):
        count += 1
        if model_from_csv[i] == LAYER_SOFTMAX:
            break
    return count

def get_new_model_number(old_model_number):
    return old_model_number+1

def get_current_model_number(latest_model):
    cur_model_num = latest_model.strip('model_')
    return int(cur_model_num)

def get_new_model(lastest_model):
    temp_new_model = get_current_model_number(lastest_model)
    new_number = get_new_model_number(temp_new_model)
    new_model = "model_"+ str(new_number)

    return new_model

def get_latest_model_list(single_model,file):
    file_name = file
    data = get_data_from_csv(file_name)
    data = format_data_without_header(data)
    lastest_model = data[INDEX_LAST][INDEX_MODEL]
    new_model = get_new_model(lastest_model)
    new_single_model = [new_model]+single_model+["Unknown","Unknown"]

    return new_single_model

def save_list_csv_rowbyrow(file_name,data_list,saving_mode = 'w'):
    mode = saving_mode
    my_file = open(file_name,mode)
    with my_file:
        writer = csv.writer(my_file)
        writer.writerows(data_list)
    return file_name

def save_topology_in_csv(file_name, data_list):
    list_of_data = data_list[:]
    csv_columns = [MODEL_TAG_HEADER]
    data_list = csv_columns + data_list

    # file_name = save_list_csv_rowbyrow(file_name,data_list,'a')
    file_name = save_list_csv_rowbyrow(file_name,data_list,'w')
    return file_name

def save_trained_model_in_csv(file_name,single_model,eval_results):
    csv_columns = MODEL_TAG_HEADER
    list_of_dict = []
    temp_dict = {}
    temp_dict[KEY_MODEL] = single_model[INDEX_MODEL]
    temp_dict[KEY_FIRST_LAYER] = single_model[INDEX_FIRST_LAYER]
    temp_dict[KEY_SECOND_LAYER] = single_model[INDEX_SECOND_LAYER]
    temp_dict[KEY_THIRD_LAYER] = single_model[INDEX_THIRD_LAYER]
    temp_dict[KEY_FORTH_LAYER] = single_model[INDEX_FORTH_LAYER]

    if isinstance(eval_results,dict):
        temp_dict[KEY_LOSS] = eval_results['loss']
        temp_dict[KEY_ACCURACY] = eval_results['accuracy']
    elif isinstance(eval_results,list):
        temp_dict[KEY_LOSS] = eval_results[0]
        temp_dict[KEY_ACCURACY] = eval_results[1]

    list_of_dict.append(temp_dict)

    print("\n####################### FINISIH TRAINING MODEL: ",temp_dict['Model'], " : #########################")
    print('\n\n\n')

    try:
       with open(file_name, 'a') as csvfile:
           writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
           if os.stat(file_name).st_size == 0 : # ONLY write ROW Hedaer when file is empty
              writer.writeheader()
           for data in list_of_dict:
               writer.writerow(data)
    except IOError:
       print("I/O error")
