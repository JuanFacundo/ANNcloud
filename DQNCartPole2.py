import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' #este para que no se queje de round off errors
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import gymnasium as gym
import random
from rl.agents import DQNAgent
#from rl.policy import BoltzmannQPolicy
#from rl.memory import SequentialMemory
import matplotlib.pyplot as plt





env = gym.make('CartPole-v1')
states = env.observation_space.shape[0]
actions = env.action_space.n
actions