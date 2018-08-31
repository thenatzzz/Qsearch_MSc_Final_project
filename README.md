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
1.
