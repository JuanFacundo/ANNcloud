from multiprocessing import reduction
import os

#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1' #este para que no se queje de round off errors

import time #to estimate the running time of the learning algorithm
from collections import deque, namedtuple

import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt             #drawing plots
import imageio
import logging # Suppress warnings from imageio
import tensorflow as tf

from pyvirtualdisplay import Display
from torch import relu
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam


#set up a virtual display to render the Lunar Lander env
#Display(visible=0, size=(840, 480)).start();

#set the random seed for TensorFlow
tf.random.set_seed(seed = 42)

MEMORY_SIZE = 100_000       #size of memory buffer
GAMMA = 0.995               #discount factor
ALPHA = 1e-3                #adams starting learning rate
batch_size = 64             #mini batch size ammount
NUM_STEPS_FOR_UPDATE = 4    #perform a learning update efery C time steps

env = gym.make('LunarLander-v2',render_mode="rgb_array")
############ACTION SPACE###########
#   0 => do nothing
#   1 => Fire right engine
#   2 => Fire main engine
#   3 => Fire left engine

#############STATE SPACE###########
#   array(x, y, v_x, v_y, angle, v_angle, L_leg, R_leg)
#   the step(action) method returns the (state after action), reward(float), terminated(bool), truncated(bool), info+(dict)

Qnetwork = Sequential([
    Input(shape = (8,)),
    Dense(units = 64, activation = 'relu'),
    Dense(units = 64, activation = 'relu'),
    Dense(units = 4)
])


mirrorQnet = Sequential([
    Input(shape = (8,)),
    Dense(units = 64, activation = 'relu'),
    Dense(units = 64, activation = 'relu'),
    Dense(units = 4)
])

optimizer = Adam(learning_rate = ALPHA)







def getLoss(exps, gamma, Qnet, sameQnet):
    """ 
    Calculates the loss.
    
    Args:
      exps: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
      Qnet: (tf.keras.Sequential) Keras model for predicting the q_values
      sameQnet: (tf.keras.Sequential) Keras model for predicting the targets
          
    Returns:
      loss: (TensorFlow Tensor(shape=(0,), dtype=int32)) the Mean-Squared Error between
            the y targets and the Q(s,a) values.
    """

    s, a, r, s_next, isDone = exps  #unfold
    
    maxQsa = tf.reduce_max(sameQnet(s_next), axis=-1)

    #set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
    isDone = tf.cast(isDone,tf.float32)
    r = tf.cast(r,tf.float32)
    y_targets = r + (1 - isDone)*gamma*maxQsa

    #get the q values and reshape it to match the y_targets
    q_values = Qnet(s)
    q_values = tf.gather_nd(q_values, tf.stack([ tf.range(q_values.shape[0]),
                                                tf.cast(a, tf.int32) ], axis=1))
    

    loss = MSE(y_targets,q_values)

    return loss




@tf.function
def Qlearn(exps, gamma):
    """
    Updates the weights of the Q networks.
    
    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
    
    """

    # calculate the loss
    with tf.GradientTape() as tape:
        loss = getLoss(exps, gamma, Qnetwork, mirrorQnet)


    # get the gradiendts of the loss with respect to the wights
    grads = tape.gradient(loss, Qnetwork.trainable_variables)

    # update the weights of the Qnet
    optimizer.apply_gradients(zip(grads, Qnetwork.trainable_variables))

    



def getA(eps, Qvalues) -> int:
    """
    Returns the best action with probabilitu (1-epsilon) otherwise
    a random action with probability epsilon to ensure exploration
    """
    
    if np.random.random() < eps:
        return env.action_space.sample()
    else:
        return int(np.argmax(Qvalues))


def checkUpdate(t, Nsteps, buff) -> bool:
    if len(buff) < batch_size:
        return False
    else:
        if t%4 == 0:
            return True
        else:
            return False




def getExps(buff):      #ojo que deque es double ended queue
    #batch_size = 64

    #random.shuffle(buff)

    exp = buff.popleft()
    mini = exp
    mini = mini._replace(state = np.array([mini.state]))
    mini = mini._replace(action = np.array([mini.action]))
    mini = mini._replace(reward = np.array([mini.reward]))
    mini = mini._replace(nextState = np.array([mini.nextState]))
    mini = mini._replace(done = np.array([mini.done]))

    buff.append(exp)

    for n in range(batch_size - 1):
        exp = buff.popleft()
        mini = mini._replace(state = np.append(mini.state,[exp.state],axis=0))
        mini = mini._replace(action = np.append(mini.action,[exp.action],axis=0))
        mini = mini._replace(reward = np.append(mini.reward,[exp.reward],axis=0))
        mini = mini._replace(nextState = np.append(mini.nextState,[exp.nextState],axis=0))
        mini = mini._replace(done = np.append(mini.done,[exp.done],axis=0))

        buff.append(exp)

    return mini


def newEpsilon(eps):
    return max(0.05,eps*0.9992)

    
    
    

def watchme(idx):
    logging.getLogger().setLevel(logging.ERROR)
    name = "./videos/LunarVideo" + idx + ".mp4"
    with imageio.get_writer(name, fps=30) as video:        #open video file till closed when "with" indentation ends
        state = env.reset()
        state = state[0]
        frame = env.render()
        video.append_data(frame)

        while True:#for n in range(1,300):
            stateQn = np.array([state])
            Qvals = Qnetwork(stateQn)
            action = getA(0, Qvals)

            state, _ , isDone, isGone, _ = env.step(action)

            isDone = isDone or isGone

            frame = env.render()
            video.append_data(frame)

            if isDone:
                break
    return








######################################################################################

start = time.time()


N_eps = 6000        # number of episodes
maxSteps = 1000     # max number of steps in each episode

pointHist = []      # total point history
good_count = 0

N_poits = 100       # number of total points to use for averaging
epsilon = 1.0       # init eps value for ε-greedy policy

# Create mem buffer D with capacity N
memBuff = deque(maxlen=MEMORY_SIZE)

# Store experiences as named tuples
Experience = namedtuple("Experience", field_names=["state", "action", "reward", "nextState", "done"])

# Set the target network weights equal to the Qnet weights
mirrorQnet.set_weights(Qnetwork.get_weights())


print()
print()
watchme("0")
for i in range(N_eps):

    # Reset the environment to the initial state and get the initial state
    state = env.reset()
    state = state[0]
    totalPts = 0

    for t in range(maxSteps):

        # From the current state S choose an action A using an ε-greedy policy
        stateQn = np.array([state])#state[0][None,:] #state needs to be the right shape for Qnet
        Qvals = Qnetwork(stateQn)
        action = getA(epsilon, Qvals)

        # Take action A and receive reward R and the next state S'
        nextState, r, isDone, isGone , _ = env.step(action)

        isDone = isDone or isGone
        # Store experience touple (S,A,R,S') in the memory buffer.
        memBuff.append(Experience(state, action, r, nextState, isDone))

        # Only update the network every NUM_STEPS_FOR_UPDATE time steps
        update = checkUpdate(t, NUM_STEPS_FOR_UPDATE, memBuff)

        if update:
            #Sample random mini-batch of experience tuples (S,A,R,S') from buffer
            experiences = getExps(memBuff)

            # Set the y targets, perform a gradient descent step, and update the network weights.
            Qlearn(experiences, GAMMA)
            # update the wights of the target Q'net
            mirrorQnet.set_weights(Qnetwork.get_weights())
            #mirrorQnet = clone_model(Qnetwork)

        state = nextState.copy()
        totalPts += r

        if isDone:
            break
            
    pointHist.append(totalPts)
    av_latest_points = np.mean(pointHist[-N_poits:])

    epsilon = newEpsilon(epsilon)

    print(f"\rEpisode {i+1} | Epsilon: {epsilon} | Total point average of the last {N_poits} episodes: {av_latest_points:.2f}", end="")

    if (i+1) % N_poits == 0:
        print(f"\rEpisode {i+1} | Epsilon: {epsilon} | Total point average of the last {N_poits} episodes: {av_latest_points:.2f}")
        watchme(str((i+1)//100))

    # We will consider that the environment is solved if we get an
    # average of 200 points in the last 100 episodes.
    if av_latest_points >= 200.0:
        good_count += 1
        if good_count > 10:
            print(f"\n\nEnvironment solved in {i+1} episodes!")
            Qnetwork.save('lunar_lander_model.h5')
            watchme("_end")
            break
        
tot_time = time.time() - start

print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")





