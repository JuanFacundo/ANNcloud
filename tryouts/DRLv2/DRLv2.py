from os import close
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math as m
import numpy.random as rand
import matlab.engine

from collections import deque, namedtuple
import tensorflow as tf
from torch import relu
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam


buffSize = 100_000
gamma = 0.995
alpha = 1e-3
batchSize = 64
nSteps4new = 10
Ts = 0.01           #Ts is the sampling rate of the PID controller and step rate of the environment




Qnetwork = Sequential([
    Input(shape = (4,)), # velocity, acceleration, P, I...
    Dense(units = 64, activation = 'relu'),
    Dense(units = 64, activation = 'relu'),
    Dense(units = 5)    # (-dP,-dI,0,+dP,+dI)
])


mirrorQnet = Sequential([
    Input(shape = (4,)), # velocity, acceleration, P, I...
    Dense(units = 64, activation = 'relu'),
    Dense(units = 64, activation = 'relu'),
    Dense(units = 5)    # (-dP,-dI,0,+dP,+dI)
])

optimizer = Adam(learning_rate = alpha)





class wavestar:
    """
    This environment class needs the following libs:
import numpy as np
import math as m
import matlab.engine
    """

    def __init__(self):
        self.s = np.zeros((6,1))
        self.fex = np.empty((1,0))
        self.n = 0
        self.T = 50.0
        self.mat = matlab.engine.start_matlab()
        self.v = [0,0,0]
        self.offset = 0


    def step(self,a):
        A = np.array([[0.983329384610631,   0.067032686254995,  -0.008347685040526,   0.005419272928689,  -0.000084042631869,  -0.000139264426848],
                  [-0.066455996022472,   0.982721387153875,  -0.017477434782007,  -0.012378428315362,  -0.000575018474091,  -0.003929207828498],
                  [0.000095853466910,  -0.015852023881767,   0.956134876560460,   0.059224807667859,  -0.000135393452137,  -0.000072846781128],
                  [0.013893787815686,  -0.002385487366262,  -0.058896514205176,   0.956652487938596,  -0.000144021904874,  -0.001322716362629],
                  [0.000188912489446,   0.000132394683352,  -0.000005724645588,  -0.000646375447684,   0.980969781935656,   0.122394581065072],
                  [-0.000112978776872,  -0.000080844506017,   0.000403046644253,  -0.000025652159388,  -0.122394566897739,   0.980969552295999]])


        B = np.array([[0.000912105410274],
                  [0.010335553369290],
                  [-0.000745433513005],
                  [0.003621876296653],
                  [0.000574107069643],
                  [0.001703845577975]])


        C = np.array([-0.153996431865825,   0.702064666021535,   0.016367212972300,   0.102420910774998,  -0.076520390394529,  -0.126027549778853])
    
        self.s = np.dot(A,self.s) + B*(a+self.fex[self.n+self.offset])

        y = np.dot(C,self.s)

        self.v[2] = self.v[1]
        self.v[1] = self.v[0]
        self.v[0] = y[0]

        estimAccel = (-self.v[2]+4*self.v[1]-3*self.v[0]) / (-2*Ts)

        r = -self.v[0]*a*Ts

        self.n+=1

        if self.n > 1000:
            isDone = True
        else:
            isDone = False

        
        if np.abs(self.v[0]) > 15:
            isGone = True
            r = r - 100
        else:
            isGone = False

        return [y[0], estimAccel], r, isDone, isGone


    def reset(self,seed):
        self.s = np.zeros((6,1))
        self.n = 0
        self.offset = np.random.randint(0,3999)
        #self.fex = self.waveGen(seed,0.15/2,1.3,3.3)
        self.v = [0,0,0]
        return self.v[0]

    def waveGen(self,seed,Hs,Tw,gamma):
        print(self.mat.eval('pwd'))
        W2F = self.mat.load('W2F','SYS_ok','tau_f')

        W2F_model = W2F['SYS_ok']
        W2F_advance = W2F['tau_f']
        w = np.double([x * 0.01 for x in range(0,5001)])

        Fe = self.mat.freqresp(W2F_model,w)
        Fe = Fe[0][0]

        for n in range(0,len(w)):
            Fe[n] = Fe[n] * self.mat.exp(1j*w[n]*W2F_advance)

        dt = Ts
        t_end = self.T
    
        Fex = self.mat.whitenoiseWave(dt,t_end,Hs,Tw,1.0,w,Fe,seed)
        Fex = np.array(Fex)
        Fex = Fex[0]

        self.fex = Fex
        
        return
    
    def tooLoud(self):
        self.loudFlag = 1



class PIDcontroller:
    """
    This is the controller that interacts with the environment.
    Theres also a sideways interaction allowed for the learning module.
    """

    def __init__(self):
        """
        Structure is defined as:
         - 0: Proportional
         - 1: PI
         - 2: PD
         - 3: PID
        """
        self.struc = 0
        self.Kp = 0
        self.Ki = 0
        self.Kd = 0
        self.last = 0
        self.I = 0
        self.dt = 0.01

    def setStruc(self,struccer):
        if struccer > 0 and struccer < 4:
            self.struc = struccer
        else:
            self.struc = 0

    def getAct(self,state):
        D = 0
        if self.struc >= 2:
            D = (state - self.last)*self.dt*self.Kd

        preI = 0
        if self.struc == 1 or self.struc == 3:
            preI = self.I + state*self.dt*self.Ki

        P = state*self.Kp

        u = P + self.I + D

        if np.abs(u) < 12:
            self.I = preI
        else:
            ws.tooLoud()
            if u > 12:
                u = 12
            if u < -12:
                u = -12

        return u
    
    def setK(self, action):
        if self.struc == 0:
            """
            0: -dP
            1: 0
            2: +dP 
            """
            if action == 0:
                self.addKp(-0.05)
                return
            elif action == 1:
                return
            elif action == 2:
                self.addKp(0.05)
                return
            else:
                print("Error: PID action rejected!")
                return
        if self.struc == 1:
            """
            0: -dI
            1: -dP
            2: 0
            3: dP
            4: dI
            """
            if action == 0:
                self.addKi(-0.5)
                return
            elif action == 1:
                self.addKp(-0.1)
                return
            elif action == 2:
                return
            elif action == 3:
                self.addKp(0.1)
                return
            elif action == 4:
                self.addKi(0.5)
                return
            else:
                print("Error: PID action rejected!")
                return
        else:
            print("You've reached a dead end. Please reconsider.")
            return


    def setKp(self,P):
        self.Kp = P

    def addKp(self,dP): #pausa lo hago logaritmico?
        self.Kp = self.Kp + dP

    def setKi(self,I):
        self.Ki = I

    def addKi(self,dI): #pausa lo hago logaritmico?
        self.Ki = self.Ki + dI

    def setKd(self,D):
        self.Kd = D

    def addKd(self,dD): #pausa lo hago logaritmico?
        self.Kd = self.Kd + dD


    def getKp(self):
        return self.Kp
    
    def getKi(self):
        return self.Ki
    
    def getKd(self):
        return self.Kd
    
    def resetID(self):
        self.last = 0
        self.I = 0




def getLoss(exps, gamma, Qnet, sameQnet):

    s, a, r, s_next, isDone = exps    #unfold

    maxQsa = tf.reduce_max(sameQnet(s_next), axis=-1)

    isDone = tf.cast(isDone, tf.float32)
    r = tf.cast(r,tf.float32)
    y_targets = r + (1 - isDone)*gamma*maxQsa

    q_values = Qnet(s)

    q_values = tf.gather_nd(q_values, tf.stack([ tf.range(q_values.shape[0]),
                                                tf.cast(a, tf.int32)   ],axis=1))

    loss = MSE(y_targets, q_values)

    return loss


@tf.function
def Qlearn(exps, gamma):

    with tf.GradientTape() as tape:
        loss = getLoss(exps, gamma, Qnetwork, mirrorQnet)

    grads = tape.gradient(loss, Qnetwork.trainable_variables)

    optimizer.apply_gradients(zip(grads, Qnetwork.trainable_variables))


def getA(eps, Qvalues) -> int:

    if np.random.random() < eps:
        return np.random.randint(0,5)
    else:
        return int(np.argmax(Qvalues))
    

def checkUpdate(t,Nsteps,buff) -> bool:
    if len(buff) < batchSize:
        return False
    else:
        if t%Nsteps == 0:
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

    t = np.linspace(0,Ts*maxSteps,maxSteps)

    Fe = ws.fex

    a = np.zeros((np.size(t,0)))
    y = np.zeros((np.size(t,0)))
    r = np.zeros((np.size(t,0)))
    kp = np.zeros((np.size(t,0)))
    ki = np.zeros((np.size(t,0)))

    y[0] = ws.reset(1)

    PIDctrl.resetID()
    PIDctrl.setKp(2.107)
    PIDctrl.setKi(-48.046)

    kp[0] = PIDctrl.getKp()
    ki[0] = PIDctrl.getKi() / 10

    for k in range(1,maxSteps):

        stateQn = np.array([state])
        Qvals = Qnetwork(stateQn)
        #action = getA(eps,Qvals)
        action = int(np.argmax(Qvals))
        a[k] = action

        PIDctrl.setK(action)

        kp[k] = PIDctrl.getKp()
        ki[k] = PIDctrl.getKi() / 10

        u = PIDctrl.getAct(-y[k-1])

        nextStateRAW, r[k], isDone, isGone = ws.step(u)

        r[k] = r[k] + r[k-1]

        isDone = isDone or isGone

        y[k] = nextStateRAW[0]

        if k == maxSteps-1 or isDone:
            plt.figure(figsize=(8, 5))
            plt.plot(t[:k],Fe[:k],label="Fe")
            plt.plot(t[:k],y[:k],label="v")
            plt.plot(t[:k],r[:k],label="R")
            plt.plot(t[:k],kp[:k],label="Kp")
            plt.plot(t[:k],ki[:k],label="Ki")
            plt.plot(t[:k],a[:k],label="a")

            name = "DRLv1_" + idx + ".png"
            
            plt.show()
            plt.savefig(name)
            # arbitrary space here?
            plt.close('all')
            return













#### Process Object ####
ws = wavestar()
# This is only part of the environment the DQN is interacting with.
# Therefor, the action space and the state space do not reflect what the DQN needs or gives.
ws.waveGen(1,0.15/2,1.3,3.3)
ws.reset(1) #the number is the seed for the wave generation.




#### Controller Object ####
PIDctrl = PIDcontroller()
# This is what the DQN actually modifies. This in combo with the wavestar instance makes the complete environment.
PIDctrl.setStruc(1) #PI


######################################################################################################
######################################################################################################



Nepis = 6000
maxSteps = 1000

pointHist = []
goodCount = 0

averWindow = 100
eps = 1.0


memBuff = deque(maxlen = buffSize)


Experience = namedtuple("Experience", field_names=["state","action","reward","nextState","done"])


mirrorQnet.set_weights(Qnetwork.get_weights())





t = np.linspace(0,Ts*1000,1000)

Fe = ws.fex

PIDctrl.resetID()
PIDctrl.setKp(2.107)
PIDctrl.setKi(-48.046)

u = np.zeros((np.size(t,0)))
y = np.zeros((np.size(t,0)))
r = np.zeros((np.size(t,0)))
R = np.zeros((np.size(t,0)))

for n in range(1,np.size(t,0)):

    u[n] = PIDctrl.getAct(-y[n-1])
    aux, r[n],_,_ = ws.step(u[n])
    y[n] = aux[0]
    R[n] = R[n-1] + r[n]

fig, ax = plt.subplots()
ax.plot(t[:1000],Fe[:1000])
ax.plot(t,y)
ax.plot(t,R)

plt.show()

print()
print()

for i in range(Nepis):

    PIDctrl.resetID()
    PIDctrl.setKp(2.107)
    PIDctrl.setKi(-48.046)

    v = ws.reset(1)
    state = np.array([-v,0,PIDctrl.getKp(),PIDctrl.getKi()])
    totalPts = 0

    for t in range(maxSteps):

        stateQn = np.array([state])
        Qvals = Qnetwork(stateQn)
        action = getA(eps, Qvals)

        PIDctrl.setK(action)

        u = PIDctrl.getAct(-v)

        nextStateRAW, r, isDone, isGone = ws.step(u)
        nextState = np.array([nextStateRAW[0], nextStateRAW[1], PIDctrl.getKp(), PIDctrl.getKi()])

        isDone = isDone or isGone

        memBuff.append(Experience(state,action,r,nextState,isDone))

        update = checkUpdate(t, nSteps4new, memBuff)

        if update:

            experiences = getExps(memBuff)

            Qlearn(experiences, gamma)

            mirrorQnet.set_weights(Qnetwork.get_weights())

        v = nextStateRAW[0]
        state = nextState.copy()

        totalPts +=r

        if isDone:
            break

    pointHist.append(totalPts)
    averageLastPts = np.mean(pointHist[-averWindow:])

    eps = newEpsilon(eps)

    print(f"\rEpisode {i+1} | Epsilon: {eps:.5f} | Total point average of the last {averWindow} episodes: {averageLastPts:.2f}",end="")

    if (i+1) % averWindow == 0:
        print(f"\rEpisode {i+1} | Epsilon: {eps:.5f} | Total point average of the last {averWindow} episodes: {averageLastPts:.2f}")
        watchme(str((i+1)//100))
        ws.waveGen(((i+1)//100),0.15/2,1.3,3.3)


    if averageLastPts >= 6.0:
        goodCount += 1
        if goodCount > 100:
            print(f"\n\nEnvironment solved in {i+1} episodes!")
            Qnetwork.save('DRLv1.h5')

            break

















"""


t = np.linspace(0,Ts*1000,1000)

Fe = ws.fex

y = np.zeros((np.size(t,0)))
r = np.zeros((np.size(t,0)))
R = np.zeros((np.size(t,0)))
I = 0
for n in range(1,np.size(t,0)):

    u = PIDctrl.getAct(-y[n-1])
    y[n], _, r[n] = ws.step(u)
    R[n] = R[n-1] + r[n]





fig, ax = plt.subplots()
ax.plot(t[:1000],Fe[:1000])
ax.plot(t,y)
ax.plot(t,R)

plt.show()



"""


