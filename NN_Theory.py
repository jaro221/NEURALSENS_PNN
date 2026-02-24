# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 06:54:24 2025

@author: jarom
Funding:  EU NextGenerationEU through the Recovery and Resilience Plan for Slovakia under the project NEURALSENS 09I05-03-V02-00058
"""

import numpy as np


"""
preparing the new NN theory
Input: constants
Weigts: functions (polynoms)
Output: values (current) 
"""

"""
weight, one layer: (4 neurons, 16 params)
    
    ax3+bx2+cx+d   ax3+bx2+cx+d
    ax3+bx2+cx+d   ax3+bx2+cx+d

"""

# Activation functions and derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)  # Derivative of sigmoid (used in backprop)

def ReLU(x):
    x[x<0]=0 
    return x 

def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9))

def f1 (X):
    return(1.2*(X**2)-0.8*X+2)

# Generate synthetic training data
np.random.seed(0)
X = np.random.rand(100, 2)  # 100 samples, 2 features
X[:,:]=0.05
X2P=np.arange(0,100)/100
X2=f1(X2P)
y = (X[:, 0] + X[:, 1] > 1).astype(int).reshape(-1, 1)  # Label: 1 if sum > 1 else 0



# Neural Network parameters
input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.1
epochs = 10000

# Initialize weights and biases
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Training loop
for epoch in range(epochs):
    # Forward Pass

        
    Z1 = np.dot(X, W1) + b1  # Hidden layer input
    A1 = ReLU(Z1)         # Hidden layer activation
    Z2 = np.dot(A1, W2) + b2  # Output layer input
    A2 = ReLU(Z2)         # Output prediction

    # Compute Loss (Mean Squared Error)
    loss = np.mean((y - A2) ** 2)

    # Backpropagation
    dA2 = -(y - A2)  # Derivative of loss w.r.t. A2
    dZ2 = dA2 * A2  # Backprop through sigmoid
    dW2 = np.dot(A1.T, dZ2) / len(X)  # Gradient of W2
    db2 = np.sum(dZ2, axis=0, keepdims=True) / len(X)

    dA1 = np.dot(dZ2, W2.T)  # Backprop to hidden layer
    dZ1 = dA1 * A1;
    dW1 = np.dot(X.T, dZ1) / len(X)  # Gradient of W1
    db1 = np.sum(dZ1, axis=0, keepdims=True) / len(X)

    # Update weights and biases using Gradient Descent
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Final loss
print(f"Final Loss: {loss:.4f}")

# Test prediction
test_sample = np.array([[0.8, 0.6]])  # Example input
hidden = sigmoid(np.dot(test_sample, W1) + b1)
output = sigmoid(np.dot(hidden, W2) + b2)
print(f"Test Prediction: {output}")







































