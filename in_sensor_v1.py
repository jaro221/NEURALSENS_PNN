# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 09:23:43 2025

@author: jarom
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import tensorflow as tf
import keras
from keras import layers
from keras.models import Model



def temp_generation(temp,param,sigma):
    
    len_data=len(temp)
    volts=(3*param*np.arange(len_data))/len_data +0.2  
    diff = np.random.normal(0, sigma, len_data)
    return volts+diff




#################################################################################################################################################
""" Synthetic data """
#################################################################################################################################################

temp =40*((np.arange(10000))/10000)+10

input1=temp_generation(temp,0.95,0.05)
input2=temp_generation(temp,0.71,0.07)
input3=temp_generation(temp,0.58,0.02)



plt.figure(figsize=(11,8))
plt.scatter(temp,input1,c="red",label="Input 1")
plt.scatter(temp,input2,c="black",label="Input 2")
plt.scatter(temp,input3,c="blue", label="Input 3")
plt.title("Synthetic data for temperature (n=10 000)",fontsize=17)

plt.ylabel("Voltage [V] ", fontsize=14)
plt.xlabel("Temperature [°C] ", fontsize=14)
plt.grid(True, linewidth=1, linestyle='--', alpha=0.5)
plt.legend(loc=(1.05,0.80),fontsize=15)
plt.ylim(0,3.5)
plt.xlim(5,55)
plt.savefig("D:/Projekty/2024_APVV/Images/synthetic_data_v2.png",bbox_inches='tight',dpi=700)
plt.show()




##################################################################################################################################################♠
""" Neural network V1"""
##################################################################################################################################################♠

model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(3,)))
model.add(layers.Dense(4,activation = 'relu'))
model.add(layers.Dense(4,activation = 'relu'))
model.add(layers.Dense(1))

      
x_train_pre=(np.vstack([input1,input2,input3]).T)  
x_train=(np.vstack([input1,input2,input3]).T)/3.3

shuffled_indices = np.random.permutation(len(x_train))
y_train=temp/100

x_train=x_train[shuffled_indices]
y_train=y_train[shuffled_indices]
temp_s=temp[shuffled_indices]
arr_sort=np.argsort(shuffled_indices)


model.compile(loss=keras.losses.MeanSquaredError(), optimizer= 'adam', metrics=["accuracy"])
history = model.fit(x_train, y_train, batch_size=64, epochs=100)

training_loss = history.history['loss']


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

yout=model.predict(x_train).flatten()

plt.figure(figsize=(11,8))
plt.plot(training_loss,c="black",label="Network_v1")
plt.title("Training loss network_v1",fontsize=17)
plt.ylabel("Loss ", fontsize=14)
plt.xlabel("Epoch ", fontsize=14)
plt.savefig("D:/Projekty/2024_APVV/Images/net_v1.png",bbox_inches='tight',dpi=700)
plt.show()

plt.figure(figsize=(11,8))
plt.scatter(temp_s,(yout-y_train)*100,c="black",label="error for Network_v1")
plt.title("Error for network_v1",fontsize=17)
plt.ylabel("Error in [°C]", fontsize=14)
plt.xlabel("Temperature ", fontsize=14)
plt.ylim(-2.5,2.5)
plt.savefig("D:/Projekty/2024_APVV/Images/error_net_v1.png",bbox_inches='tight',dpi=700)
plt.show()

plt.figure(figsize=(11,8))
plt.scatter((x_train[:,0]*3.3)[arr_sort],(y_train*100)[arr_sort],c="black",label="True values")
plt.scatter((x_train[:,0]*3.3)[arr_sort],(yout*100)[arr_sort],c="red",label="Predicted values")
plt.title("Prediction for network_v1",fontsize=17)
plt.ylabel("Predicted temperature [V] ", fontsize=14)
plt.xlabel("Voltage [V] ", fontsize=14)
plt.ylim(5,55)
plt.legend(loc=(1.05,0.80),fontsize=15)
plt.show()
plt.savefig("D:/Projekty/2024_APVV/Images/pred_net_v1.png",bbox_inches='tight',dpi=700)














