"""
Sean McDonald

10/31/16

Neural Network Version 3.0

A slightly more advanced neural network when compared to V2. Implements a very
basic backward propagation algorithm in order to reach the result. Uses Standard
Gradient Descent with a variety of Alpha values that modify the size of the weight
adjustment. This has two layers, and implements a hidden layer to mess around
with the hidden layer size to see if it has any effect.
"""

import time
import numpy as np

# The various alpha values
alphas = [.0001, .001, .01, .1, 1, 10, 100]

# The hidden layer. This changes up the number of values in the hidden layer
# (the layer(s) between the input and output nodes) to see if it has any effect
# on the results.

hiddenLayerSize = 8

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
    t0= time.time()
    print("Alpha:")
    print(alpha)
    
                       
# This ensures that the random numbers are seeded, meaning the random
# number distribution will be the same each time training occurs
    np.random.seed(1)


# This is the matrix for the weights of the first synapse. In this case,
# it is the only synapse between the input and output nodes. It is a matrix,
# where the initial weight is generated randomly. Improvements to this method
# will come later. THe matrix must be equivalent to (input nodes, output nodes).
    synapse0 = 2*np.random.random((5,8)) - 1
    synapse1 = 2*np.random.random((5,5)) - 1
    synapse2 = 2*np.random.random((5,5)) - 1
    synapse3 = 2*np.random.random((5,5)) - 1

# The neural network itself, ran an arbitrary number of times
    for i in range(100000):
    
    # Data is propogated forward, with the 0th layer being the input, and the
    # 1st layer having the data be put through a sigmoid function with the
    # input data and the weight of synapse 0
    
        layer0 = inputData
        layer1 = sigmoid(np.dot(layer0,synapse0))
        layer2 = sigmoid(np.dot(layer1,synapse1))
        layer3 = sigmoid(np.dot(layer2,synapse2))
        layer4 = sigmoid(np.dot(layer3,synapse3))
    
    #Find the error by comparing layer 2 to the proper output data
    
        layer4Error = outputData - layer4
        
        if ((i% 10000) == 0):
            print("Error after "+str(i)+" iterations:" + 
            str(np.mean(np.abs(layer4Error))))
            
            
        layer4Delta = layer4Error * sigmoid_derivative(layer4)
        layer3Error = layer4Delta.dot(synapse3.T)
        layer3Delta = layer3Error * sigmoid_derivative(layer3)
        layer2Error = layer3Delta.dot(synapse2.T)
    
    # Multiply the error with the slope of the sigmoid function, as the layers
    # have already been set with the sigmoid function, the derivative value
    # is set to true.
    
        layer2Delta = layer2Error * sigmoid_derivative(layer2)
        
    # Use the delta value to calculate the error for layer 1
        
        layer1Error = layer2Delta.dot(synapse1.T)
        
    # Now calculate the layer 1 Delta value for updating the synapses
        
        layer1Delta = layer1Error * sigmoid_derivative(layer1)
    
    # Actually apply the changes to the weights. Subtraction is here rather
    # than addition because the goal is to reduce to "slope" to 0, rather than
    # to constantly update the weights.
    
    # The subtraction works, but it is heavily dependent on the alpha value
    # If the alpha value is not good, the output will be crazy
    # Need to check error, then determine proper alpha value
    
    #It seems that addition is the best option for getting correct results
    
        synapse0 += alpha * (layer0.T.dot(layer1Delta))
        synapse1 += alpha * (layer1.T.dot(layer2Delta))
        synapse2 += alpha * (layer2.T.dot(layer3Delta))
        synapse3 += alpha * (layer3.T.dot(layer4Delta))
        
    print("Time for process:")
    print(time.time() - t0)
    print("Output after training:")
    print(layer2)
