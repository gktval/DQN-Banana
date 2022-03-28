import torch
import math
from dqn_enums import dqn_types

class Config(object):
    def __init__(self):
        #Device type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #Type of model
        self.model = dqn_types.dqn

        #Random seed
        self.seed = 0

        #algorithm control
        self.USE_NOISY_NETS=False
        self.USE_PRIORITY_REPLAY=False
        
        #Multi-step returns
        self.N_STEPS = 1

        #epsilon variables
        self.epsilon_start = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 30000
        self.epsilon_by_frame = lambda frame_idx: self.epsilon_final + (self.epsilon_start - self.epsilon_final) * math.exp(-1. * frame_idx / self.epsilon_decay)

        #misc agent variables
        self.GAMMA=0.99 # discount factor
        self.LR=1e-4 # learning rate 
        self.TAU = 1e-3 # for soft update of target parameters
        self.layer_size = 512 # size of the hidden layer

        #memory
        self.TARGET_NET_UPDATE_FREQ = 1000
        self.EXP_REPLAY_SIZE = 100000
        self.BATCH_SIZE = 32
        self.PRIORITY_ALPHA=0.6
        self.PRIORITY_BETA_START=0.4
        self.PRIORITY_BETA_FRAMES = 100000

        #Noisy Nets
        self.SIGMA_INIT=0.5

        #Learning control variables
        self.LEARN_START = 10000
        self.MAX_FRAMES=100000
        self.UPDATE_FREQ = 1 # how often to update the network
