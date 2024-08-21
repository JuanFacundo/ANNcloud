import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input


mirrorQnet = Sequential([
    Input(shape = (8,)),
    Dense(units = 64, activation = 'relu'),
    Dense(units = 64, activation = 'relu'),
    Dense(units = 4)
])

s = np.array([1,2,3,4,5,6,7,8])
p = np.array([s])
#p = np.append(p,[s],axis=0)
p = np.array(np.append(p,[s],axis=0))
print(p)
print(s)
print(mirrorQnet(p))