MSc Final Project: Q-Search - designing CNN architectures using Reinforcement Learning

Scope:
1. Limited numbers of State (Layer) = 4
2. Limited numbers of Action (Different types of layers) = 16
3. Possible CNN layers: Convolutional, Maximum Pooling and Softmax Layer

Main Algorithm and Techniques:
1. Q-Learning with Q-table
2. Epsilon-Greedy Strategy: balancing between Exploration and Exploitation
3. Experience Replay: a memory for the agent
    Different ways to use Experience Replay
    3.1 Not using Experience Replay
    3.2 Update Q-table by sampling models from Experience Replay after the agent
        finishes training new model.
    3.4 Update Q-table by sampling models from Experience Replay in periodic manner
        such as every 100 episodes

Dataset:
1. MNIST
2. CIFAR-10

Experiments:
1. Random Search: in MAIN folder by using RANDOM_TOPOLOGY.py then train according
   to specified datasets
   1.1 MNIST
   1.2 CIFAR-10
2. Layerwise Search: in LAYERWISE_SEARCH folder by usng MAIN.py then train according
   to specified datasets
   2.1 MNIST
   2.2 CIFAR-10
3. Q-Search: in MAIN folder, to run using MAIN.py
           : the parametes of Q-Learning can be adjusted in QLEARNING.py
             such as number of episodes, experience replay updating techniques,
             number of model to be sampled from, Q-Discount rate, Q-Learning rate,
             epsilon schedule
   3.1 MNIST
   3.2 CIFAR-10

PS. the parameters for CNN can be adjusted in TRAIN_MODEL_MNIST/CIFAR10.py such as
    training steps, batch size, data augmentation

To Visualize the Results:
1. VISUALIZE_HEATMAP: to see how Q-table changes overtime
2. VISUALIZE_MODEL_CHART: to see how topologies the agent has successfully found overtime
                          going in what direction.
3. VISUALIZE_DIST: to see how distribution of model: mean, median, mode, max

REQUIREMENTS:
1. Tensorflow platform for MNIST, CIFAR-10
2. Keras platform for CIFAR-10
3. Seaborn for visualization
4. (Optional) ffmpeg for visualizing heatmap png files
5. (Optional) using GPU setup to speed up training

HOW TO USE:
1. RANDOMLY generate topology for epsilon = 1 (using RANDOM_TOPOLOGY.py)
2. train file from 1. with pre_train_model_mnist/cifar10 in TRAIN_MODEL_MNIST/CIFAR-10.py
3. using file from 1. as experience replay
4. start Q-learning

Debugging:
1. If training MNIST dataset, get this error message:
    "NotFoundError (see above for traceback): Key conv2d_2/bias not found in checkpoint"
    Solution: check whether there is finished model located in saved folder,
              If there is, please move old finished model to different folder.
2. careful in using file_name

NOTE: there are two different TRAIN files for two different datasets because
      MNIST is implemented in pure Tensorflow and CIFAR-10 is implemented in Keras
      which uses Tensorflow as backend.
