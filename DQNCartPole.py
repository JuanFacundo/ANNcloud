#pip install gymnasium[classic_control]
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collection import namedtupe,deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make("CartPole-v1")

is_ipython = 'inline' in matplotlib.get_backend
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device("cuda" if torch.cude.is_available() else "cpu")



Transition = namedtupe('Transition',('state','action','next_state','reward'))

class ReplayMemory(object):

    def __init__(self,capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self,*args):
        self.memory.append(Transition(*args))
        """Save a transition"""

    def sample(self,batch_size):
        return random.sample(self,memory,batch_size)
    
    def __len__(self):
        return len(self.memory)
    




class DQN(nn.Module):
    #multi-layer perceptron with three layers

    #n_observations is input(state of environment) to the network

    #n_actions - number of possible actions in the environment
    def __init__(self, n_observations, n_actions):
        super(DQN,self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128,128)
        self.layer3 = nn.Linear(128,n_action)

    #take and pass through the 3 layers of the network
    def forward(self,x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
#number of transition samples from the repay buffer
BATCH_SIZE = 128
#Gamma is the discount factor
GAMMA = 0.99 
EPS_START = 0.9 #the start value of epsilon
EPS_END = 0.05 #the end value of epsilon
EPS_DECAY = 1000 #Controls the rate of the exponential decay
TAU = 0.005 #Update rate of the target network
LR = 1e-4 #Learning rate of the AdamW optimizer (the W stands for the weights added to the dimention of the learned parameters of the networks)




#get n of actions from gym action space
n_actions = env.action_space.n

#get the number of state observations
state,info = env.reset()

#number of features in the state
n_observations = len(state)

#target net is initialized with the same weight as 'plicy_net'
policy_net = DQN(n_observations,n_actions).to(device)
target_net = DQN(n_observations,n_actions,).to(device)
target_net.load_state_dict(policy_net.state_dict())

#optimizer - AdamW
optimizer = optim.AdamW(policy_net.parameters(),lr=LR, amsgrad)
#stores agent experience, which will be used for training
memory = ReplayMemory(10000)

#keeps track of n of steps taken by the agent
steps_done = 0

#inputs current state - outputs an action
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
    steps_done +=1
    if sample > eps_threshold:
        with torch.no_grad():
            #t.max(1) will return th largest column value
            #second column on max result is index of where
            #was found, so we pick action with the larger value
            return policy_net(state).max(1)[1].view(1,1)
    else:
        return torch.tensor([[env.action_space.sample()]],dtype = torch.long)
    

#keeps track of the duration of each episode
episode_durations = []

#function used to visualize the training of the DQN
def plot_duration(show_result=False):
    plt.figure(1)
    duration_t = torch.tensor(episode_durations,dtype = torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training')
    plt.xlable('Episode')
    plt.ylable('Duration')
    plt.plot(duration_t.numpy())

    #shows the 100-episode moving average of the durations
    if(len(duration_t) >= 100):
        means = duration_t.unfold(0,1000,1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99),means))
        plt.plot(means.numpy())

    plt.pause(0.001) #pause so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            displat.clear_output(wait=True)
        else:
            display.display(plt.gcf())








def optimize_mode():
    if len(memory) < BATCH_SIZE:
        return
    transition = memory.sample(BATCH_SIZE)
    #converts batch array of Transitions to Transition of batch arrays
    batch = Transition(*zip(*transition))

    non_final_mask = torch.tensor(tuple(map(lambda s: s in batch.next_state)))