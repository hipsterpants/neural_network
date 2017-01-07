"""
Sean McDonald

10/6/16

Neural Network Version 1.0

The most basic neural network, no backpropagation algorithm. Simply updates
weights and attempts to get near correct output. Only works properly with
extremely small data sets (and I mean extremely small). There are only two
layers, the input nodes and the output nodes. Will identify patterns, even
if the patterns aren't there.

This is a 2-layer neural network.

"""

import numpy as np
import time

# (Manual) Input data in the form of a numpy array (similar to a matrix)
# Each row is a training example while each column is an input node
# In this instance the first value in each set is the output
inputData = np.array([[0,0,0],
                      [1,0,1],
                      [0,1,0],
                      [1,1,0]])

                      
# (Manual) Output data in the form of a numpy array (similar to a matrix)
# Each row is a training example, with each row corresponding to the row
# in the input.
# Each column is an output node. Meaning that the value should line up with the
# value in the first column of each training input.
outputData = np.array([[1,0],
                       [0,1],
                       [1,0],
                       [1,1]])
                       
# This ensures that the random numbers are seeded, meaning the random
# number distribution will be the same each time training occurs
np.random.seed(1)

# This is the matrix for the weights of the first synapse. In this case,
# it is the only synapse between the input and output nodes. It is a matrix,
# where the initial weight is generated randomly. Improvements to this method
# will come later. The matrix must be equivalent to (input nodes, output nodes).
synapse0 = 2*np.random.random((3,2)) - 1

# Standard Sigmoid function for each individual neuron.

def sigmoid(x):
    return 1/(1+np.exp(-x))
    
# Returns the derivative of the sigmoid function for use after the sigmoid
# neurons have already been created.
    
def sigmoid_derivative(x):
    return x*(1-x)
    

# The neural network itself, ran an arbitrary number of times
for iteration in range(100000):
    t0 = time.time();
    
    # Data is propogated forward, with the 0th layer being the input, and the
    # 1st layer having the data be put through a sigmoid function with the
    # input data and the weight of synapse 0. In this case, layer 1 is the
    # output layer.
    
    layer0 = inputData
    layer1 = sigmoid(np.dot(layer0,synapse0))
    
    #Find the error by comparing layer 1 to the proper output data
    
    layer1Error = outputData - layer1
    
    if ((iteration% 10000) == 0):
        print("Error after "+str(iteration)+" iterations:" + 
        str(np.mean(np.abs(layer1Error))))
    
    # Multiply the error with the slope of the sigmoid function, as the layers
    # have already been set with the sigmoid function, the derivative value
    # is set to true.
    
    layer1Delta = layer1Error * sigmoid_derivative(layer1)
    
    # Actually apply the changes to the weights
    
    synapse0 += np.dot(layer0.T, layer1Delta)
    
# Prints the output after however many iterations. The closer to the initial
# output, the more accurate the neural network is!
print("Time taken:")
print(time.time() - t0)
print("Output:")
print(layer1)