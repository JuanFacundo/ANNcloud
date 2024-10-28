from multiprocessing import reduction
import os

#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1' #este para que no se queje de round off errors

from telnetlib import EXOPL
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

#from LunarLander import Experience, Qnetwork



buffSize = 100_000
gamma = 0.995
alpha = 1e-3
batchSize = 64
nSteps4new = 4



env = gym.make('CartPole-v1', render_mode="rgb_array")
##############  Action Space  ##############
#   0 => Push cart to the left
#   1 => Push cart to the right

##############  State Space   ##############
#   array(x, v, a, w)





Qnetwork = Sequential([
    Input(shape = (4,)),
    Dense(units = 32, activation = 'relu'),
    Dense(units = 32, activation = 'relu'),
    Dense(units = 2)
])

mirrorQnet = Sequential([
    Input(shape = (4,)),
    Dense(units = 32, activation = 'relu'),
    Dense(units = 32, activation = 'relu'),
    Dense(units = 2)
])

optimizer = Adam(learning_rate = alpha)








def getLoss(exps, gamma, Qnet, sameQnet):
    
    s, a, r, s_next, isDone = exps  #unfold

    maxQsa = tf.reduce_max(sameQnet(s_next), axis=-1)

    isDone = tf.cast(isDone, tf.float32)
    r = tf.cast(r,tf.float32)
    y_targets = r + (1 - isDone)*gamma*maxQsa

    q_values = Qnet(s)

    q_values = tf.gather_nd(q_values, tf.stack([ tf.range(q_values.shape[0]),
                                              tf.cast(a, tf.int32)  ],axis=1))

    loss = MSE(y_targets, q_values)

    return loss




@tf.function
def Qlearn(exps,gamma):
    
    with tf.GradientTape() as tape:
        loss = getLoss(exps, gamma, Qnetwork, mirrorQnet)

    grads = tape.gradient(loss, Qnetwork.trainable_variables)

    optimizer.apply_gradients(zip(grads, Qnetwork.trainable_variables))






def getA(eps,Qvalues) -> int:

    if np.random.random()   < eps:
        return env.action_space.sample()
    else:
        return int(np.argmax(Qvalues))
    

def checkUpdate(t,Nsteps,buff) -> bool:
    if len(buff) < batchSize:
        return False
    else:
        if t%4 == 0:
            return True
        else:
            return False
        

def getExps(buff):
    exp = buff.popleft()
    mini = exp
    mini = mini._replace(state = np.array([mini.state]))
    mini = mini._replace(action = np.array([mini.action]))
    mini = mini._replace(reward = np.array([mini.reward]))
    mini = mini._replace(nextState = np.array([mini.nextState]))
    mini = mini._replace(done = np.array([mini.done]))

    buff.append(exp)
    
    for n in range(batchSize - 1):
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
    name = "./videos/CartVideo" + idx + ".mp4"
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







######################################################################




start = time.time()

Nepis = 6000
maxSteps = 1000

pointHist = []
goodCount = 0

averWindow = 100
eps = 1.0

memBuff = deque(maxlen=buffSize)

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "nextState", "done"])


mirrorQnet.set_weights(Qnetwork.get_weights())




print()
print()
watchme("0")

for i in range(Nepis):

    state = env.reset()
    state = state[0]
    totalPts = 0


    for t in range(maxSteps):

        stateQn = np.array([state])
        Qvals = Qnetwork(stateQn)
        action = getA(eps, Qvals)

        nextState, r, isDone, isGone, _ = env.step(action)
        isDone = isDone or isGone

        memBuff.append(Experience(state,action,r,nextState,isDone))

        update = checkUpdate(t, nSteps4new, memBuff)

        if update:

            experiences = getExps(memBuff)

            Qlearn(experiences, gamma)

            mirrorQnet.set_weights(Qnetwork.get_weights())


        state = nextState.copy()

        totalPts += r

        if isDone:
            break

    pointHist.append(totalPts)
    averageLastPts = np.mean(pointHist[-averWindow:])

    eps = newEpsilon(eps)

    print(f"\rEpisode {i+1} | Epsilon: {eps} | Total point average of the last {averWindow} episodes: {averageLastPts:.2f}", end="")

    if (i+1) % averWindow == 0:
        print(f"\rEpisode {i+1} | Epsilon: {eps} | Total point average of the last {averWindow} episodes: {averageLastPts:.2f}")
        watchme(str((i+1)//100))

    if averageLastPts >= 200.0:
        goodCount += 1
        if goodCount > 100:
            print(f"\n\nEnvironment solved in {i+1} episodes!")
            Qnetwork.save('cartandpole_model.h5')
            watchme("_end")
            break

tot_time = time.time() - start

print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")