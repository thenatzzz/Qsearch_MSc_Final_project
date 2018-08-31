MSc Final Project: Q-Search- designing CNN architectures using Reinforcement Learning

Scope:
1. Limited numbers of State (Layer) = 4
2. Limited numbers of Action (Different types of layers) = 16
3. Possible CNN layers: Convolutional, Maximum Pooling and Softmax Layer

Main Algorithms and Techniques:
1. Q-Learning with Q-table
2. Epsilon-Greedy Strategy: balancing between Exploration and Exploitation
3. Experience Replay: memory for the agent
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
             such as number of episodes, updating techm
   3.1 MNIST
   3.2 CIFAR-10

PS. the parameters for CNN can be adjusted in TRAIN_MODEL_MNIST/CIFAR10.py  
