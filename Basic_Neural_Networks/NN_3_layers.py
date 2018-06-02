import numpy as np

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)
    
class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.w1   = np.random.rand(self.input.shape[1],4) 
        self.w2   = np.random.rand(4,4)
        self.w3   = np.random.rand(4,1) 
        self.y          = y
        self.out     = np.zeros(self.y.shape)

    def feedforward(self):
        z1 = np.dot(self.input, self.w1)
        self.a1 = sigmoid(z1)
        z2 = np.dot(self.a1, self.w2)
        self.a2 = sigmoid(z2)
        z3 = np.dot(self.a2, self.w3)
        self.out = sigmoid(z3)

    def backprop(self):
        #application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        
        error = (self.y - self.out)
        o_delta = (error * sigmoid_derivative(self.out))
        
        a2_error = np.dot(o_delta, self.w3.T)
        a2_delta = a2_error * sigmoid_derivative(self.a2)
        
        a1_error = np.dot(a2_delta, self.w2.T)
        a1_delta = a1_error * sigmoid_derivative(self.a1)
        
        d_w3 = np.dot(self.a2.T,  o_delta)
        d_w2 = np.dot(self.a1.T,  a2_delta)
        d_w1 = np.dot(self.input.T,  a1_delta)

        # update the weights with the derivative (slope) of the loss function
        self.w1 += d_w1
        self.w2 += d_w2
        self.w3 += d_w3
        
X = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])

y = np.array([[0],[1],[1],[0]])
nn = NeuralNetwork(X,y)


for i in range(1500):
    nn.feedforward()
    nn.backprop()
    
print(nn.out)
    
