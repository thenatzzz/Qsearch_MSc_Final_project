# from QLEARNING import run_q_learning
from HELPER_FUNCTION import get_data_from_csv, format_data_without_header
from RANDOM_TOPOLOGY import get_random_topology
from TRAIN_MODEL_CIFAR10 import pre_train_model_svhn
# from tmpTRAIN_MODEL_MNIST import pre_train_model_mnist


#PBS -l select=1:ncpus=8:mpiprocs=4

if __name__ == "__main__":

    #Get random topologies then save to csv file
    # random_topology_file  = 'test_random_topology.csv'
    # num_model = 1500
    # file_name = get_random_topology(num_model, random_topology_file)
    # print(file_name)
    # pre_train_model_svhn(file_name)


    file = "model_svhn_dict.csv"
    pre_train_model_svhn(file)

    '''
    #Run Q-learning to find best topology
    file_name = "fixed_model_dict.csv"
    # file_name = "bad_model.csv"
    # file_name = "biased_dict.csv"
    # file_name = "COMPLETE_CIFAR10.csv"
    # data = open_file(file_name)
    data = get_data_from_csv(file_name)
    data = format_data_without_header(data)
    dataset = "mnist"
    best_topology = run_q_learning(data,dataset)
    print("best_topology: ", best_topology)
    # accuracy, loss = to_verify_model(best_topology)
    '''

    #Get random topologies then save to csv file
    # random_topology_file  = 'test_random_topology.csv'
    # num_model = 1500
    # file_name = get_random_topology(num_model, random_topology_file)
    # print(file_name)
    # pre_train_model_cifar10(file_name)
    '''
    #Run Q-learning to find best topology
    # file_name = "fixed_model_dict.csv"
    # file_name = "bad_model.csv"
    # file_name = "biased_dict.csv"
    file_name = "COMPLETE_CIFAR10.csv"
    # data = open_file(file_name)
    data = get_data_from_csv(file_name)
    data = format_data_without_header(data)
    # print(data[0])
    dataset = "cifar10"
    best_topology = run_q_learning(data,dataset)
    print("best_topology: ", best_topology)
    # accuracy, loss = to_verify_model(best_topology)
    '''
