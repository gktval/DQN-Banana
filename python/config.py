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

        #Categorical Params
        self.ATOMS = 51
        self.V_MAX = 10
        self.V_MIN = -10

        #Quantile Regression Parameters
        self.QUANTILES=51

        #DRQN Parameters
        self.SEQUENCE_LENGTH=8

        #data logging parameters
        self.ACTION_SELECTION_COUNT_FREQUENCY = 1000


'''

#epsilon variables
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

#misc agent variables
GAMMA=0.99
LR=1e-4

#memory
TARGET_NET_UPDATE_FREQ = 1000
EXP_REPLAY_SIZE = 100000
BATCH_SIZE = 32
PRIORITY_ALPHA=0.6
PRIORITY_BETA_START=0.4
PRIORITY_BETA_FRAMES = 100000

#Noisy Nets
SIGMA_INIT=0.5

#Learning control variables
LEARN_START = 10000
MAX_FRAMES=1000000

'''