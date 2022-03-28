import numpy as np
import random
from dqn_enums import dqn_types
from memory_replay import PrioritizedReplay, ReplayBuffer
import model
from dqn_enums import dqn_types

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, config):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        self.Q_updates = 0
        self.state_size = state_size
        self.action_size = action_size
        self.seed = config.seed
        self.total_reward = 0
        self.n_step = config.N_STEPS
        self.buffer_size = config.EXP_REPLAY_SIZE
        self.gamma = config.GAMMA
        self.tau = config.TAU
        self.batch_size = config.BATCH_SIZE
        self.update_every = config.UPDATE_FREQ
        self.lr = config.LR
        self.priority_replay = config.USE_PRIORITY_REPLAY
        self.is_noisy = config.USE_NOISY_NETS
        self.layer_size = config.layer_size
        self.device = config.device
        randomSeed = random.seed(self.seed)
        
        # Q-Network
        if config.model == dqn_types.dqn:
            self.qnetwork_local = model.DQN(state_size, action_size, self.seed, self.is_noisy).to(self.device)
            self.qnetwork_target = model.DQN(state_size, action_size, self.seed, self.is_noisy).to(self.device)
        elif config.model == dqn_types.ddqn:
            self.qnetwork_local = model.DDQN(state_size, action_size, self.layer_size, self.n_step, self.seed, self.is_noisy).to(self.device)
            self.qnetwork_target = model.DDQN(state_size, action_size, self.layer_size, self.n_step, self.seed, self.is_noisy).to(self.device)
        elif config.model == dqn_types.ddqn_c51:
            self.qnetwork_local = model.DDQN_C51(state_size, action_size, self.layer_size, self.n_step, self.seed, self.is_noisy, self.atoms, self.v_max, self.v_min).to(self.device)
            self.qnetwork_target = model.DDQN_C51(state_size, action_size, self.layer_size, self.n_step, self.seed, self.is_noisy, self.atoms, self.v_max, self.v_min).to(self.device)
        elif config.model == dqn_types.dueling_dqn:
            self.qnetwork_local = model.Dueling_QNetwork(state_size, action_size, self.layer_size, self.n_step, self.seed, self.is_noisy).to(self.device)
            self.qnetwork_target = model.Dueling_QNetwork(state_size, action_size, self.layer_size, self.n_step, self.seed, self.is_noisy).to(self.device)
        elif config.model == dqn_types.dueling_c51:
            self.qnetwork_local = model.Dueling_C51Network(state_size, action_size, self.layer_size, self.n_step, self.seed, self.is_noisy, self.atoms, self.v_max, self.v_min).to(self.device)
            self.qnetwork_target = model.Dueling_C51Network(state_size, action_size, self.layer_size, self.n_step, self.seed, self.is_noisy, self.atoms, self.v_max, self.v_min).to(self.device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        if self.priority_replay:
            self.memory = PrioritizedReplay(self.buffer_size,self.batch_size, self.seed, self.device, self.gamma, self.n_step)
        else:
            self.memory = ReplayBuffer(self.action_size,self.buffer_size, self.batch_size, self.seed, self.device)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done): 
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:          
                experiences = self.memory.sample()
                loss = self.learn(experiences)            
                self.Q_updates += 1
        

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        numpyState = torch.from_numpy(state)
        state = numpyState.float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()).astype(np.int32)
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones, idx, weights = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma**self.n_step * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        if self.priority_replay:
            td_error =  Q_targets - Q_expected
            loss = (td_error.pow(2)*weights).mean().to(self.device)       
        else:
            loss = F.mse_loss(Q_expected, Q_targets) 
        
        # Minimize the loss
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(),1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau) 

        if self.priority_replay: # update per priorities
            self.memory.update_priorities(idx, abs(td_error.data.cpu().numpy()))

        return loss.detach().cpu().numpy()      

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
