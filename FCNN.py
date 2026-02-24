# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 13:01:14 2025

@author: jarom

Funding:  EU NextGenerationEU through the Recovery and Resilience Plan for Slovakia under the project NEURALSENS 09I05-03-V02-00058
"""

import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
from keras import layers
from keras.models import Model

# Activation function: ReLU and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Define a simple fully connected neural network for regression
class SimpleFCNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with random values (using He initialization for ReLU)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        # Hidden layer computations
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)
        # Output layer (linear output for regression)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2
    
    def compute_loss(self, y_pred, y_true):
        # Mean Squared Error loss
        loss = np.mean((y_pred - y_true) ** 2)
        return loss
    
    def backward(self, X, y_true, y_pred, learning_rate):
        m = X.shape[0]  # number of examples
        
        # Compute gradient for the output layer
        dz2 = (2 / m) * (y_pred - y_true)  # gradient of loss w.r.t. z2
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        # Backprop into the hidden layer
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * relu_derivative(self.z1)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights and biases using gradient descent
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
    
    def train(self, X, y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            # Forward pass: compute predictions
            y_pred = self.forward(X)
            # Compute the loss
            loss = self.compute_loss(y_pred, y)
            # Backward pass: update parameters
            self.backward(X, y, y_pred, learning_rate)
            # Print loss every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

# Example usage:

np.random.seed(0)

# Create sample data: 100 examples, 2 features each
X = np.random.rand(100, 2)

# Define a target for regression: y = 3*x1 + 2*x2 + noise
y = 3 * X[:, 0:1] + 2 * X[:, 1:2] + np.random.randn(100, 1) * 0.1

# Initialize the network: 2 inputs, 10 hidden neurons, 1 output
model = SimpleFCNN(input_size=2, hidden_size=10, output_size=1)

# Train the model
#model.train(X, y, epochs=1000, learning_rate=0.01)

# Test the model on a new example
test_example = np.array([[0.5, 0.2]])
prediction = model.forward(test_example)
prediction_1=model.forward(X)
dif=y-prediction_1

print("Test Prediction:", prediction)


##########################################################################################################################

def f1 (X):
    return(1.2*(X**2)-0.8*X+2)
def f2 (X):
    return(0.4*X-0.1)

def poly_fun(W,X):
    """
    input weight - matrix (2,2)
    input X2 correspond to the values of environment (in this case - tempareture for testing)
    everyone weight correspond to hte one sensor with polynomial characteristic - for test ony quadratic approach

    W= WM2   
    X= k1yr
    """
    X2O=np.zeros((W.shape[0],W.shape[1], 1000,1),dtype="float32")
    for idx in range(W.shape[0]):
        for idxx in range(W.shape[1]):
            poly=np.poly1d(W[idx,idxx,:])
            res=poly(X)
            X2O[idx,idxx,:,:]=poly(X)
    return X2O

def poly_dot(I,W,XM):
    """    
    I=k1w
    W= W1  
    XM= k1
    """
    
    X2O=np.zeros((1000,W.shape[1]),dtype="float32")
    for idx in range(1000):
        for idx2 in range(W.shape[1]):
            pre_res=0
            for idx3 in range(W.shape[0]):
                res=I[idx,idx2]*W[idx3,idx2]*XM[idx3,idx2,idx,0]
                pre_res+=res
            X2O[idx,idx2]=pre_res
    return X2O


def poly_dot_rev(I,dz,WMM):
    """  
    X2, dW1, k1, WM1
    I=X2
    dz= dz2 
    WMM=WM2
    """
    WMrev=np.zeros(WMM.shape)
    for i in range(WMM.shape[0]):
        for ii in range(WMM.shape[1]):
            poltest=np.polyfit(I.flatten(),dz[:,ii],WMM.shape[2]-1) # I or X-values (environment, and error as Y-values)
            WMrev[i,ii]=poltest
    return WMrev

def sum_w(inp_W,KB):
    """
    inp_W=k1
    KB = kb1
    """
    new_iw=np.zeros(inp_W.shape)
    for idx in range(1000):
        new_iw[:,:,idx,0] = inp_W[:,:,idx,0]+KB
    return new_iw

def temp_generation(temp,param,sigma):
    
    len_data=len(temp)
    volts=(param*np.arange(len_data))/len_data +0.2  
    diff = np.random.normal(0, sigma, len_data)
    return volts+diff






X1=np.random.rand(1000, 2)  # 100 samples, 2 features
X1[:,0]=0.5
X1[:,1]=0.1
X2P=np.arange(10,30,0.02)/100
temp =40*((np.arange(1000))/1000)+10

input1=temp_generation(temp,0.95,0.05)
input2=temp_generation(temp,0.71,0.07)
X2P=temp_generation(temp,0.58,0.03)

X2=np.asarray([f1(X2P)]).T
#X2=np.asarray([X2P]).T
#=np.asarray([X2P]).T
polynom1=np.poly1d([1,1])
y = np.asarray([temp/50]).T

X=X1
epochs=90
input_size=2
hidden_size=2
output_size=1
learning_rate=0.01
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
b1 = np.zeros((1, hidden_size))
b1A = np.zeros((1, hidden_size))+1.5

W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
b2 = np.zeros((1, output_size))
b2A = np.zeros((1, output_size))+2

WM1=np.zeros((2,2,4),dtype="float")
WM1 = np.random.rand(2,2,4)
WM2=np.zeros((2,1,4),dtype="float")
WM2 = np.random.rand(2,1,4)

kb1=np.zeros((W1.shape))
kb1[:,:]=1.5
kb2=np.zeros((W2.shape))
kb2[:,:]=1.5
cmap=plt.colormaps.get_cmap("jet")
hist=np.zeros(epochs,dtype="float32")
plt.figure()
all_errors=np.zeros((epochs,1000),dtype="float32")
all_pred=np.zeros((epochs,1000),dtype="float32")
y_res=np.zeros((epochs,1000),dtype="float32")
for epoch in range(epochs):
    # Forward pass: compute predictions
    """"""
    k1 = poly_fun(WM1, X2) #4x"y" or 4x"X2"
    k1w=poly_dot(X1,W1,k1)
    k1wb=k1w+b1A
    k1wbr=relu(k1wb)
    
    k2 = poly_fun(WM2, X2)
    k2w=poly_dot(k1wbr,W2,k2)
    kk2=np.dot(k1wbr,W2)
    k2wb=k2w+b2A
    y_true=y
    y_tpre=k2wb
    y_troz = k2wb - y 
    y_res[epoch,:]=y_troz[:,0]
    ###############################
    loss=np.mean((k2wb - y) ** 2)
    m = X.shape[0]
    
    dz2 = (2 / m) * (k2wb - y)
    dW2=np.dot(k1wbr.T,dz2)
    
    db2 = np.sum(dz2, axis=0, keepdims=True)
    
    dWM2= poly_dot_rev(X2, dz2, WM2)


    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * relu_derivative(k1wbr)
    dW1 = np.dot(X1.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)
    dWM1= poly_dot_rev(X2, dz1, WM1)
    """"""
    plt.plot(k2wb,c=cmap(epoch/epochs))
    all_errors[epoch,:]=y_troz[:,0]
    all_pred[epoch,:]=k2wb[:,0]
    
    # Update weights and biases using gradient descent
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    WM1= WM1 - learning_rate * dWM1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    WM2= WM2 - learning_rate * dWM2
    

    # Print loss every 100 epochs
    hist[epoch]=loss
    if epoch % 1 == 0:
        print(f"Epoch {epoch}, LOSS: {np.round(loss,5)} and ERROR: {np.round(np.mean(np.abs(y_troz)),5)}")






plt.figure()
for epoch in range(epochs):
    plt.plot(all_errors[epoch,:],c=cmap(epoch/epochs))
plt.ylim(-1,1)

plt.figure()
for epoch in range(epochs):
    plt.plot(all_pred[epoch,:],c=cmap(epoch/epochs))
plt.plot(y,c="black")
plt.ylim(0.4,0.6)



import keras

model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(2,)))
model.add(layers.Dense(2,activation = 'relu'))
model.add(layers.Dense(1))

          
model.compile(loss=keras.losses.MeanSquaredError(), optimizer= 'adam', metrics=["accuracy"])
XX=np.concatenate((X2,X2),axis=1)
history = model.fit(XX, y, batch_size=2, epochs=100)


layers_model=model.layers
W1 = model.layers[0].get_weights()[0]
B1  = model.layers[0].get_weights()[1]
W2 = model.layers[1].get_weights()[0]
B2  = model.layers[1].get_weights()[1]


y_pred=model.predict(XX)

ynn=y_pred-y
yfnn=k2wb-y
print(f"{np.round(np.mean(np.abs(ynn)),5)} and {np.round(np.mean(np.abs(yfnn)),5)}")











plt.figure()
for epoch in range(epochs):
    plt.plot(y,all_errors[epoch,:],c=cmap(epoch/epochs))
plt.title("Error for FNN")
plt.ylabel("Error")
plt.xlabel("Input_2")
plt.legend(loc=(1.05,0.80),fontsize=10)
plt.savefig("D:/Projekty/2024_APVV/PNN_25_05_21_ERROR_p2.png",bbox_inches='tight',dpi=700)


plt.figure()
for epoch in range(epochs):
    plt.plot(y,all_errors[epoch,:],c=cmap(epoch/epochs))
plt.ylim(-0.2,0.4)
plt.title("Error for FNN")
plt.ylabel("Error")
plt.xlabel("Input_2")
plt.legend(loc=(1.05,0.80),fontsize=10)
plt.savefig("D:/Projekty/2024_APVV/PNN_25_05_21_ERROR_FOCUS_p2..png",bbox_inches='tight',dpi=700)



plt.figure()
plt.plot(hist, color="orange",label="loss")
plt.title("Loss values for FNN")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(loc=(1.05,0.80),fontsize=10)
plt.savefig("D:/Projekty/2024_APVV/PNN_25_05_21_LOSS_p2..png",bbox_inches='tight',dpi=700)



# Example if arrays not defined:
# y_pred, k2wb, y, X2 = [np.random.rand(1000) for _ in range(4)]

fig, ax1 = plt.subplots()            # Main axis (left Y-axis)
ax2 = ax1.twinx()                    # Secondary Y-axis (right)
ax1.plot(np.arange(1000), y_pred, c="blue", label="ANN")
ax1.plot(np.arange(1000), k2wb, c="red", label="PNN")
ax1.plot(np.arange(1000), y, c="green", label="Temperature", linewidth=3)
ax2.plot(np.arange(1000), X2, c="lime", label="Temperature - encoded", linewidth=1)
ax1.set_title("Prediction for NNs")
ax1.set_xlabel("Input_2")
ax1.set_ylabel("Value")
ax2.set_ylabel("Encoded Temperature")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc=(1.15, 0.80), fontsize=10)
plt.savefig("D:/Projekty/2024_APVV/ANN_25_05_21_True_p2.png", bbox_inches='tight', dpi=700)


plt.figure()
plt.plot(y,y_pred,c="blue",label="ANN")
plt.plot(y,all_errors[-1,:],c="red",label="PNN")
plt.title("Error for NNs")
plt.ylabel("Error")
plt.xlabel("Input_2")
plt.legend(loc=(1.05,0.80),fontsize=10)
plt.savefig("D:/Projekty/2024_APVV/ANN_25_05_21_ERROR_p2..png",bbox_inches='tight',dpi=700)



plt.figure()
plt.plot(history.history["loss"], color="orange",label="loss")
plt.title("Loss values for ANN")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(loc=(1.05,0.80),fontsize=10)
plt.savefig("D:/Projekty/2024_APVV/ANN_25_05_21_LOSS_p2..png",bbox_inches='tight',dpi=700)



######################################################################################################
""" WM1 """
WMM=WM1
fig, axs = plt.subplots(WMM.shape[1], WMM.shape[0], figsize=(10, 8)) 
for idx in range(WMM.shape[0]):
    for idxx in range(WMM.shape[1]):
        print(WMM[idx,idxx,:])
        poly=np.poly1d(WMM[idx,idxx,:])
        res=poly(y)
        axs[idx,idxx].plot(res)

for idx in range(WMM.shape[0]):
    for idxx in range(WMM.shape[1]):
        axs[idx,idxx].set_ylim(0,2)

plt.savefig("D:/Projekty/2024_APVV/PNN_25_05_21_LAYER_1.png",bbox_inches='tight',dpi=700)




ta=np.asarray([[5,2],[3,2]])
tb=np.asarray([1,4])
tc=np.dot(ta,tb)


tb1=np.linalg.solve(ta, tc)


X2A=np.asarray([f1(X2P)]).T
X2B=np.asarray([f2(X2P)]).T

plt.plot(X2A)
plt.plot(X2B)
plt.plot(X2B-X2A)
plt.plot(X2P)


tpol=np.polyfit(X2P,X2B-X2A,2)


