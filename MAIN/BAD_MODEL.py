from HELPER_FUNCTION import get_data_from_csv, format_data_without_header,\
                            save_topology_in_csv

LOSS_COL_INDEX = -1
ACCURACY_COL_INDEX = -2

def get_bad_topology(data,threshold,factor):
    list_data = []
    for index in range(len(data)):
        if factor == "accuracy":
            if float(data[index][ACCURACY_COL_INDEX]) < threshold:
                list_data.append(data[index])
        elif factor == "loss":
            if float(data[index][LOSS_COL_INDEX]) > threshold:
                list_data.append(data[index])
    return list_data

def main():
    
    FILE_NAME = "fixed_model_dict.csv"
    FILE_NAME = "COMPLETE_CIFAR10.csv"

    data = get_data_from_csv(FILE_NAME)
    data = format_data_without_header(data)

    OUTPUT_FILE = "bad_model.csv"

    accuracy_threshold = 0.7
    bad_model = get_bad_topology(data,accuracy_threshold,"accuracy")
    # loss_threshold = 1.0
    # bad_model = get_bad_topology(data,loss_threshold,"loss")

    bad_model_file = save_topology_in_csv(OUTPUT_FILE,bad_model)
    print("Create file for bad topology: ",bad_model_file, " successfully!!")

if __name__ == "__main__":
    main()
