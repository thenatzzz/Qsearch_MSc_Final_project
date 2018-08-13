from HELPER_FUNCTION import get_data_from_csv, format_data_without_header
import statistics

INDEX_ACCURACY = -2
INDEX_LOSS = -1

def get_list(data,target):
    final_list = []
    for i in range(len(data)):
        if target == "accuracy":
            final_list.append(float(data[i][INDEX_ACCURACY]))
        else:
            final_list.append(float(data[i][INDEX_LOSS]))
    return final_list

def get_mean(data,target):
    final_list = get_list(data,target)
    mean = statistics.mean(final_list)
    return mean

def get_var(data,target):
    final_list = get_list(data,target)
    var = statistics.variance(final_list)
    return var

def get_standard_deviation(data,target):
    final_list  = get_list(data,target)
    std_dev = statistics.stdev(final_list)
    return std_dev

def get_accuracy_model(model):
    return model[INDEX_ACCURACY]

def get_loss_model(model):
    return model[INDEX_LOSS]

def get_number_model(data, target, above_or_below,criteria):
    temp_list = []
    val_criteria = 0
    if criteria == 'mean':
        val_criteria = get_mean(data,target)
    elif criteria == 'var':
        val_criteria = get_var(data,target)
    elif criteria == 'std':
        val_criteria = get_standard_deviation(data,target)
    else:
        val_criteria = criteria

    if target == 'accuracy':
        index = INDEX_ACCURACY
    elif target == 'loss':
        index = INDEX_LOSS

    for datum in data:
        if above_or_below == 'above':
            if float(datum[index]) > val_criteria:
                temp_list.append(datum)
        elif above_or_below == 'below':
            if float(datum[index]) < val_criteria:
                temp_list.append(datum)

    return len(temp_list) , temp_list

def get_best_topology(data,target):
    best_model = data[0]
    best_model_acc = get_accuracy_model(best_model)
    current_model = ""

    for i in range(len(data)):
        current_model = data[i]
        current_model_acc = get_accuracy_model(current_model)
        if current_model_acc > best_model_acc:
            best_model_acc = current_model_acc
            best_model = current_model
    # print(best_model)
    return best_model

def get_worst_topology(data,target):
    worst_model = data[0]
    worst_model_acc = get_accuracy_model(worst_model)
    current_model = ""

    for i in range(len(data)):
        current_model = data[i]
        current_model_acc = get_accuracy_model(current_model)
        if current_model_acc < worst_model_acc:
            worst_model_acc = current_model_acc
            worst_model = current_model
    # print(worst_model)
    return worst_model

def main():
    file_name = "fixed_model_dict.csv"
    file_name = "COMPLETE_CIFAR10.csv"

    #MNIST: CLEAN_CODE
    file_path = "/homes/nj2217/PROJECT/TMP5_CLEAN_CODE/FINISHED_MODEL/MNIST/CLEAN_CODE/"
    file = "fixed_model_dict.csv"
    file_name = file_path + file

    #CIFAR-10: TMP_CODE
    file_path ="/homes/nj2217/PROJECT/TMP5_CLEAN_CODE/FINISHED_MODEL/CIFAR-10/TMP_CODE/"
    file = "COMPLETE_CIFAR10.csv"
    file_name = file_path + file

    #CIFAR-10: TMP2_CLEAN_CODE
    # lost file

    #CIFAR-10: TMP3_CLEAN_CODE
    file_path ="/homes/nj2217/PROJECT/TMP5_CLEAN_CODE/FINISHED_MODEL/CIFAR-10/TMP3_CLEAN_CODE/"
    file = "COMPLETE_CIFAR10.csv"
    file_name = file_path + file

    #CIFAR-10: TMP_CODE_2
    file_path ="/homes/nj2217/PROJECT/TMP5_CLEAN_CODE/FINISHED_MODEL/CIFAR-10/TMP_CODE_2/"
    file = "COMPLETE_CIFAR10.csv"
    file_name = file_path + file

    #CIFAR-10: TMP3_CLEAN_CODE
    file_path ="/homes/nj2217/PROJECT/TMP5_CLEAN_CODE/FINISHED_MODEL/CIFAR-10/TMP2_CODE_2/"
    file = "COMPLETE_CIFAR10.csv"
    file_name = file_path + file


    target = "loss"
    target = "accuracy"

    above_or_below_target = "below"
    above_or_below_target = "above"

    criteria = "mean"
    # criteria = 0.7413

    data = get_data_from_csv(file_name)
    data = format_data_without_header(data)

    best_model = get_best_topology(data,target)
    worst_model = get_worst_topology(data,target)

    print("Std: ",get_standard_deviation(data,target))
    print("Mean: ",get_mean(data,target))
    print("Variance: ",get_var(data,target))

    num_model, list_model = get_number_model(data,target,above_or_below_target,criteria)
    print("Total number of model: ",len(data))
    print("Number of model->", num_model," that is ",above_or_below_target, " (",criteria," of ",target,")")

if __name__ == "__main__":
    main()
