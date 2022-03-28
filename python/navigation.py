from math import fabs
from pickle import TRUE
from dqn_enums import dqn_types
from unityagents import UnityEnvironment
import numpy as np
from numpy import loadtxt
import torch
from collections import deque
from dqn_agent import Agent
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config
import os

def dqn(env, config, n_episodes=800, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):   

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of actions
    action_size = brain.vector_action_space_size

    # examine the state space 
    state = env_info.vector_observations[0]
    state_size = len(state)

    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
        
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.GAMMA = 0.99
    config.LR = 1e-4
    config.TAU = 1e-3 
    config.seed = 0
    
    agent = Agent(state_size=state_size, action_size=action_size,config=config)
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state  
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)     
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

    env.close()
    return scores, scores_window, agent

def ddqn(env, config, n_episodes=800, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):   
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of actions
    action_size = brain.vector_action_space_size

    # examine the state space 
    state = env_info.vector_observations
    state_size = state

    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score  
    
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.GAMMA = 0.99
    config.LR = 1e-4
    config.TAU = 1e-3 
    config.seed = 0  
    config.layer_size = 512  

    agent = Agent(state_size=state_size, action_size=action_size,config=config)
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state  
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)     
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

    env.close()
    return scores, scores_window, agent

def plot_seaborn(array_counter, array_score, train):
    sns.set(color_codes=True, font_scale=1.5)
    sns.set_style("white")
    plt.figure(figsize=(13,8))
    fit_reg = False if train== False else True        
    ax = sns.regplot(
        np.array([array_counter])[0],
        np.array([array_score])[0],
        #color="#36688D",
        x_jitter=.1,
        scatter_kws={"color": "#36688D"},
        label='Data',
        fit_reg = fit_reg,
        line_kws={"color": "#F49F05"}
    )
    # Plot the average line
    y_mean = [np.mean(array_score)]*len(array_counter)
    ax.plot(array_counter,y_mean, label='Mean', linestyle='--')
    ax.legend(loc='upper right')
    ax.set(xlabel='# games', ylabel='score')
    plt.show()

def WatchSmartAgent(env):
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)
    agent = Agent(state_size=state_size, action_size=action_size, seed=0)
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

    for j in range(1000):
        action = agent.act(state)
                
        env_info = env.step(action)[brain_name]        # send the action to the environment
        done = env_info.local_done[0]                  # see if episode has finished
        if done:
            break 
            
    env.close()

def saveScores(filename, scores):
    dirName = 'scores'
    dirExists = os.path.isdir(dirName)
    if not dirExists:
        os.mkdir(dirName) 

    filename = os.path.join(dirName, filename) + '.txt'
    with open(filename, "w") as f:
        for s in scores:
            f.write(str(s) +"\n")

def run(isTest, dqnType):
    # please do not modify the line below
    env = UnityEnvironment(file_name="../p1_navigation/Banana_Windows_x86_64/Banana.exe")
    config = Config()
    config.USE_NOISY_NETS= False
    config.USE_PRIORITY_REPLAY=False

    config.model = dqnType
    if isTest:
        if dqnType == dqn_types.dqn:
            scores, scores_window, agent = dqn(env, config)
        elif dqnType == dqn_types.ddqn:
            scores, scores_window, agent = ddqn(env, config)
        elif dqnType == dqn_types.dueling_dqn:
            scores, scores_window, agent = ddqn(env, config)

        filename = str(dqnType.name)
        saveScores(filename, scores)       

        if np.mean(scores_window) >= 13:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(len(scores), np.mean(scores_window)))
        else:
            print('\nEnvironment failed to solve in {:d} episodes.\tAverage Score: {:.2f}'.format(len(scores), np.mean(scores_window)))

        torch.save(agent.qnetwork_local.state_dict(), filename + '_checkpoint.pth')
        plot_seaborn(np.arange(len(scores)), scores, True)

    else:
        WatchSmartAgent(env)

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def showScores():
    
    plt.figure(figsize=(13,8))   

    #Iterate through each file in saved scores
    for filename in os.listdir('scores'):
        file = os.path.join('scores', filename)
        # checking if it is a file
        if os.path.isfile(file):
            print(file)
            if not file.endswith('.txt'):
                continue
            scores = loadtxt(file, comments="#", delimiter=",", unpack=False)
            scores = moving_average(scores,50)
            iterations = np.arange(len(scores))

            sns.set(color_codes=True, font_scale=1.5)
            sns.set_style("white")
            ax = sns.lineplot(
                    np.array([iterations])[0],
                    np.array([scores])[0],
                    label=os.path.basename(file).split('.')[0],
                )
            # Plot the average line
            y_mean = [np.mean(scores)]*len(iterations)
            
            ax.legend(loc='upper left')
            ax.set(xlabel='# games', ylabel='score')        
    plt.show()

if __name__ == '__main__':
    # Set options to activate or deactivate the game view, and its speed 
    run(True, dqn_types.dqn)

    #show all scores in scores folder
    #showScores()