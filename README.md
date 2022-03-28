[//]: # (Image References)

[image1]: https://video.udacity-data.com/topher/2018/June/5b1ab4b0_banana/banana.gif "Banana Environment"

# DQN Banana Agent

![Trained Agents][image1]

This repository contains material related to Udacity's Value-based Methods course.

### Introduction

The code lead you through implementing various algorithms in reinforcement learning.  All of the code is in PyTorch (v0.4) and Python 3.
This code uses three Deep Q-Networks for training an agent to navigate (and collect bananas!) in a large, square world.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

    0 - move forward.
    1 - move backward.
    2 - turn left.
    3 - turn right.

The project environment is similar to, but not identical to the Banana Collector environment on the Unity ML-Agents GitHub page. 

### Setup

This project simulates an environment from [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents).

1.    Download the environment from one of the links below. You need only select the environment that matches your operating system:
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
		- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
		- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
		- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

2. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/gktval/DQN-banana
cd python
pip install .
```

3. Navigate to the `python/` folder. Run the file `navigation.py` found in the `python/` folder.

### Navigating the code

Running the code without any changes will start a unity session and train the DQN agent. Alternatively, you can change the agent model in the run method. The following agents are available as options:

	DQN
	DDQN
	Dueling-DQN

In the run method, you can change configuration options for the selected Deep Q-Network. This includes the use of noisy networks and priority replay buffers. The results of the training will be stored in a folder called `scores` location in the `python` folder. After running several of the DQN networks, you can replay the trained agent by changing the `isTest` variable passed into the `run()` method. You should also change filename of the checkpoint in `WatchSmartAgent` to the checkpoint you wish to replay. Furthermore, this method will display the scores of all trained agents in the `scores` folder.

### Summary
This graph shows the scores of the various trained DNQ agents used in this project. The DDQN and Dueling-DQN networks performed less well than the DQN network. Only the DQN and the DQN-noisy network reached an average targeted score of 13+ after 600 episodes. 

[image1]: https://github.com/gktval/DQN-banana/blob/main/Summary%20Results.png "Scores of trained agents within 800 games"

