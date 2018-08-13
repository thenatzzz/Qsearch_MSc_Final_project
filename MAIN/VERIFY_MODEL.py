from TRAIN_MODEL_MNIST import train_model_mnist
from TRAIN_MODEL_CIFAR10 import train_model_cifar10

def format_into_normal_form(single_model):
    FIRST_LAYER = single_model[0]
    SECOND_LAYER = single_model[1]
    THIRD_LAYER = single_model[2]
    FORTH_LAYER = single_model[3]
    temp = [FIRST_LAYER, SECOND_LAYER,THIRD_LAYER, FORTH_LAYER]
    last_layer = ["unknown", "unknown"]
    single_model = ["verified_model"]+temp + last_layer
    return single_model

def get_original_format(best_model_dict):
    tmp_list = []
    tmp_list.append(best_model_dict['Layer 1'][0])
    tmp_list.append(best_model_dict['Layer 2'][0])
    tmp_list.append(best_model_dict['Layer 3'][0])
    tmp_list.append(best_model_dict['Layer 4'][0])
    print(tmp_list)
    return tmp_list

def verify_model(single_model,dataset):
    is_verify = True

    if isinstance(single_model,dict):
        single_model = get_original_format(single_model)

    single_model = format_into_normal_form(single_model)

    if dataset =="cifar10":
        train_model_cifar10(single_model,is_verify)
    elif dataset == 'mnist':
        train_model_mnist(single_model,is_verify)
