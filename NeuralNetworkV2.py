"""
Sean McDonald

10/17/16

Neural Network Version 2.0

A slightly more advanced neural network when compared to V1. Implements a very
basic backward propagation algorithm in order to reach the result. Uses Standard
Gradient Descent with a variety of Alpha values that modify the size of the weight
adjustment.
"""

import numpy as np
import time

# The various alpha values
alphas = [.0001, .001, .01, .1, 1, 10, 100]

# Standard Sigmoid function for each individual neuron.

def sigmoid(x):
    return 1/(1+np.exp(-x))
    
# Returns the derivative of the sigmoid function for use after the sigmoid
# neurons have already been created.
    
def sigmoid_derivative(x):
    return x*(1-x)
    
# (Manual) Input data in the form of a numpy array (similar to a matrix)
# Each row is a training example while each column is an input node
# In this instance the first value in each set is the output
inputData = np.array([[0, 0, 1, 0, 1],
                      [0, 1, 1, 1, 0],
                      [1, 0, 1, 0, 0],
                      [1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 0]])
                      
# (Manual) Output data in the form of a numpy array (similar to a matrix)
# Each row is a training example, with each row corresponding to the row
# in the input.
# Each column is an output node. Meaning that the value should line up with the
# value in the first column of each training input.
outputData = np.array([[0, 1, 0, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1],
                       [1, 0, 0, 0, 0]])
                       

# The loop that ensures the neural network goes through all different alpha
# values                       
for alpha in alphas:
    t0 = time.time();
    print("Alpha:")
    print(alpha)
    print("Output after training:")
    
                       
# This ensures that the random numbers are seeded, meaning the random
# number distribution will be the same each time training occurs
    np.random.seed(1)


# This is the matrix for the weights of the first synapse. In this case,
# it is the only synapse between the input and output nodes. It is a matrix,
# where the initial weight is generated randomly. Improvements to this method
# will come later. THe matrix must be equivalent to (input nodes, output nodes).
    synapse0 = 2*np.random.random((5,5)) - 1

# The neural network itself, ran an arbitrary number of times
    for i in range(100000):
    
    # Data is propogated forward, with the 0th layer being the input, and the
    # 1st layer having the data be put through a sigmoid function with the
    # input data and the weight of synapse 0
    
        layer0 = inputData
        layer1 = sigmoid(np.dot(layer0,synapse0))
    
    #Find the error by comparing layer 1 to the proper output data
    
        layer1Error = outputData - layer1
        
        if ((i% 10000) == 0):
            print("Error after "+str(i)+" iterations:" + 
            str(np.mean(np.abs(layer1Error))))
    
    # Multiply the error with the slope of the sigmoid function, as the layers
    # have already been set with the sigmoid function, the derivative value
    # is set to true.
    
        layer1Delta = layer1Error * sigmoid_derivative(layer1)
        synapse_derivative = np.dot(layer0.T, layer1Delta)
    
    # Actually apply the changes to the weights. Subtraction is here rather
    # than addition because the goal is to reduce to "slope" to 0, rather than
    # to constantly update the weights.
    
    # The subtraction works, but it is heavily dependent on the alpha value
    # If the alpha value is not good, the output will be crazy
    # Need to check error, then determine proper alpha value
    
        synapse0 += alpha * (layer0.T.dot(layer1Delta))
    
    print(layer1)
    print("Time taken:")
    print(time.time() - t0)
        

print("Output after training:")
print(layer1)
