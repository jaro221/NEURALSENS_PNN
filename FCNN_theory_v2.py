# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 10:56:48 2025

@author: jarom
"""

import numpy as np

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


WM1=np.zeros((2,2,3),dtype="float")
WM1 = np.random.rand(2,2,3)
WM2=np.zeros((2,1,3),dtype="float")
WM2 = np.random.rand(2,1,3)

def f1 (X):
    return(1.2*(X**2)-0.8*X+2)

def poly_fun(W,X2):
    """
    input weight - matrix (2,2)
    input X2 correspond to the values of environment (in this case - tempareture for testing)
    everyone weight correspond to hte one sensor with polynomial characteristic - for test ony quadratic approach

    W= WM1   
    """
    for idx in range(W.shape[0]):
        for idxx in range(W.shape[1]):
            poly=np.poly1d(WM1[idx,idxx,:])
            res=poly(X2)
            W[idx,idxx]=poly(X2)
    return W

X1=np.random.rand(100, 2)  # 100 samples, 2 features
X1[:,:]=0.05
X2P=np.arange(0,100)/100
X2=np.asarray([f1(X2P)]).T
polynom1=np.poly1d([1,1])
y = polynom1(X2)


X=X1
epochs=10000
input_size=2
hidden_size=2
output_size=1
learning_rate=0.01
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
b2 = np.zeros((1, output_size))