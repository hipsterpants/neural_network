"""
Sean McDonald

10/31/16

Neural Network Version 4.0

A slightly more advanced neural network when compared to V3. Implements a very
basic backward propagation algorithm in order to reach the result. Uses Standard
Gradient Descent with a variety of Alpha values that modify the size of the weight
adjustment. This has two layers, and implements a hidden layer to mess around
with the hidden layer size to see if it has any effect.
This one implements the IrisFileReader methods (until I find a better way to 
do this).
"""

import numpy as np

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
                      [1, 1, 1, 1, 1]])
                      
# (Manual) Output data in the form of a numpy array (similar to a matrix)
# Each row is a training example, with each row corresponding to the row
# in the input.
# Each column is an output node. Meaning that the value should line up with the
# value in the first column of each training input.
outputData = np.array([[1, 0, 1],
                       [1, 1, 0],
                       [0, 0, 1],
                       [1, 1, 1]])
                       

# The loop that ensures the neural network goes through all different alpha
# values                       
for alpha in alphas:
    print("Alpha:")
    print(alpha)
    print("Output after training:")
    print(layer1)
    
                       
# This ensures that the random numbers are seeded, meaning the random
# number distribution will be the same each time training occurs
    np.random.seed(1)


# This is the matrix for the weights of the first synapse. In this case,
# it is the only synapse between the input and output nodes. It is a matrix,
# where the initial weight is generated randomly. Improvements to this method
# will come later. THe matrix must be equivalent to (input nodes, output nodes).
    synapse0 = 2*np.random.random((5,3)) - 1

# The neural network itself, ran an arbitrary number of times
    for i in range(100000):
    
    # Data is propogated forward, with the 0th layer being the input, and the
    # 1st layer having the data be put through a sigmoid function with the
    # input data and the weight of synapse 0
    
        layer0 = inputData
        layer1 = sigmoid(np.dot(layer0,synapse0))
    
    #Find the error by comparing layer 1 to the proper output data
    
        layer1Error = outputData - layer1
    
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
    
        synapse0 -= alpha * (layer0.T.dot(layer1Delta))
        
print("Output after training:")
print(layer1)