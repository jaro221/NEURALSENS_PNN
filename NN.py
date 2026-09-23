# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:20:54 2024

@author: jarom
"""

import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.models import Model


x_train=np.asarray([[0.1,0.1,0.2],[0.2,0.3,0.2],[0.4,0.5,0.4]])
y_train=np.asarray([0.1,0.2,0.4])

model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(3,)))
model.add(layers.Dense(5,activation = 'relu'))
model.add(layers.Dense(5,activation = 'relu'))
model.add(layers.Dense(1))

          
model.compile(loss=keras.losses.MeanSquaredError(), optimizer= 'adam', metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=2, epochs=100, validation_split=0.2)

layers_model=model.layers
W1 = model.layers[0].get_weights()[0]
B1  = model.layers[0].get_weights()[1]
W2 = model.layers[1].get_weights()[0]
B2  = model.layers[1].get_weights()[1]
W3 = model.layers[2].get_weights()[0]
B3  = model.layers[2].get_weights()[1]

y_pred=model.predict(x_train)

L1A=np.dot(x_train,W1)
L1B=L1A+B1
L1=np.copy(L1B)
L1[L1B<0]=0

L2A=np.dot(L1,W2)
L2B=L2A+B2
L2=np.copy(L2B)
L2[L2B<0]=0

L3A=np.dot(L2,W3)
L3B=L3A+B3
L3=np.copy(L3B)
L3[L3B<0]=0





















