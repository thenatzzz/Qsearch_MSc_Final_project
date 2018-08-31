from QLEARNING import run_q_learning
from HELPER_FUNCTION import get_data_from_csv, format_data_without_header
from RANDOM_TOPOLOGY import get_random_topology
from TRAIN_MODEL_CIFAR10 import pre_train_model_cifar10
from TRAIN_MODEL_MNIST import pre_train_model_mnist
from VERIFY_MODEL import verify_model, get_original_format

if __name__ == "__main__":

    '''NOTE: there are two different TRAIN file for different dataset because
          MNIST is implemented in pure Tensorflow and CIFAR-10 is implemented in Keras
          which uses Tensorflow as backend. '''


    '''
    ############################################################################
    ######## Get random topologies then save to csv file #######################
    ######## Random Search     (MNIST)                   #######################
    ############################################################################
    INPUT_FILE_NAME_RANDOM_TOPO  = 'test_random_topology.csv'
    NUM_MODEL = 1500
    OUTPUT_FILE_NAME = "mnist_model.csv" # act as experience replay
    INPUT_FILE_NAME = get_random_topology(NUM_MODEL, INPUT_FILE_NAME_RANDOM_TOPO)
    print(INPUT_FILE_NAME)
    ############################################################################
    ########### Train model in the csv file ####################################
    ############################################################################
    pre_train_model_mnist(INPUT_FILE_NAME,OUTPUT_FILE_NAME)
    '''

    '''
    ###########################################################################
    ############  Run Q-learning to find best topology ########################
    ############            MNIST                      ########################
    ###########################################################################
    file_name = "mnist_model.csv"           # memeory replay file
    data = get_data_from_csv(file_name)
    data = format_data_without_header(data)
    DATASET = "mnist"
    best_topology = run_q_learning(data,DATASET)
    print("Higest values at the end of training: ", best_topology)
    # verify_model(best_topology,DATASET)
    '''

    '''
    ############################################################################
    ######## Get random topologies then save to csv file #######################
    ######## Random Search     (CIFAR-10)                #######################
    ############################################################################
    INPUT_FILE_NAME_RANDOM_TOPO  = 'test_random_topology.csv'
    NUM_MODEL = 1500
    OUTPUT_FILE_NAME = "cifar10_model.csv"
    INPUT_FILE_NAME = get_random_topology(NUM_MODEL, INPUT_FILE_NAME_RANDOM_TOPO)
    print(INPUT_FILE_NAME)
    pre_train_model_cifar10(INPUT_FILE_NAME,OUTPUT_FILE_NAME)
    '''

    '''
    ###########################################################################
    ############  Run Q-learning to find best topology ########################
    ###########################################################################
    file_name = "cifar10_model.csv"
    data = get_data_from_csv(file_name)
    data = format_data_without_header(data)
    DATASET = "cifar10"
    best_topology = run_q_learning(data,DATASET)
    print("Higest values at the end of training: ", best_topology)
    verify_model(best_topology,DATASET)
    '''

    '''
    ###########################################################################
    ############  To train some model                  ########################
    ###########################################################################
    model = ['c_1','c_8','c_12','c_9']
    DATASET = 'cifar10'
    # DATASET = 'mnist'
    verify_model(model, DATASET)
    '''
