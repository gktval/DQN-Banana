### REPORT
Implementation
This project implements and compares 3 deep Q-networks in respect to the Unity’s Banana environment. The three networks tested are DQN, Double DQN, and Dueling DQN. Several other parameters tested include the use of noisy networks and priority replay buffers.
To setup the project, follow the instructions in the README file. 
To implement the DQN network, simply indicate the type of network to run in the `init__()` method of navigation.py file.  In the `run()` method, the configuration of the network can be changed, including the GAMMA and the LR (learning rate).  One can also designate the use of noisy networks and priority replay buffers. 

### Results
The results of the training will be stored in a folder called `scores` location in the `python` folder. After running several of the DQN networks, you can replay the trained agent by changing the `isTest` variable passed into the `run()` method. You should also change filename of the checkpoint in `WatchSmartAgent` to the checkpoint you wish to replay. Furthermore, this method will display the scores of all trained agents in the `scores` folder.

Only the DQN and DQN-noisy agents were able to receive an average reward of +13 over 100 episodes. This average was accomplished after 600 episodes.  With these agents, two hyperparameters where changed; GAMMA of .99 and a LR of 1e-4. 
The plot of rewards can be found here:
https://github.com/gktval/DQN-banana/blob/main/Summary%20Results.png
The scores of the weights from the DQN agent can be found here:
https://github.com/gktval/DQN-banana/blob/main/python/scores/dqn_checkpoint.pth

### Future work
Future work should include using a few of the networks not included in this submission, including distributed DQNs and rainbow networks. 


