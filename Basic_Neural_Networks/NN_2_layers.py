import numpy as np

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)
    
class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4) 
        self.weights2   = np.random.rand(4,1)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        z1 = np.dot(self.input, self.weights1)
        self.layer1 = sigmoid(z1)
        z2 = np.dot(self.layer1, self.weights2)
        self.output = sigmoid(z2)

    def backprop(self):
        #application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        error = (self.y - self.output)
        d2 = 2 * error * sigmoid_derivative(self.output)
        d_weights2 = np.dot(self.layer1.T, d2)
        
        d1 = np.dot(2* error * sigmoid_derivative(self.output), self.weights2.T)
        d_weights1 = np.dot(self.input.T,  d1 * sigmoid_derivative(self.layer1))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        
X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
y = np.array([[0],[1],[1],[0]])
nn = NeuralNetwork(X,y)

import pdb
pdb.set_trace()

for i in range(1500):
    nn.feedforward()
    nn.backprop()

print(nn.output)