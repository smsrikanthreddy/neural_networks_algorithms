import numpy as np
from sklearn.linear_model import LogisticRegressionCV
import sklearn.datasets
import matplotlib.pyplot as plt

np.random.seed(3)
X, y = sklearn.datasets.make_moons(200, noise=0.20)
print('X shape is:-', np.shape(X),'y shape is:-', y.shape)
#plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
#plt.show()

# %% 3
# Train the logistic rgeression classifier
clf = LogisticRegressionCV()
clf.fit(X, y)

# %% 4
# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot below.
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


# %% 12
# Plot the decision boundary
#plot_decision_boundary(lambda x: clf.predict(x))
#plt.title("Logistic Regression")


num_examples = len(X) # training set size
nn_input_dim = 2 # input layer dimensionality
nn_output_dim = 2 # output layer dimensionality

# Gradient descent parameters (I picked these by hand)
learning_rate = 0.01 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength
steps = 1000


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def  tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def tanh_dericative(x):
    return (1 - np.power(x, 2))

def neural_network_model(nn_hdim):

    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim)
    b2 = np.zeros((1, nn_output_dim))

    for i in range(0, steps):

        # Forward propagation to calculate our predictions
        z1 = np.dot(X, W1) + b1
        a1 = tanh(z1)
        z2 = np.dot(a1, W2) + b2
        probs = softmax(z2)

        # Calculating the loss
        #loss_cost =  -np.sum(np.log(probs) * y_label)
        corect_logprobs = -np.sum(np.log(probs[range(num_examples), y]))

        #delta3 = probs - y
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = np.dot(a1.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)

        delta2 = np.dot(delta3, W2.T) * tanh_dericative(a1)
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Gradient descent parameter update
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2

        if i % 100 == 0:
            print("Loss after iteration %i: %f" % (i,corect_logprobs))
            #corect_logprobs = 0

        model_parameters = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    return model_parameters

# Helper function to predict an output (0 or 1)
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = np.dot(x, W1) + b1
    a1 = tanh(z1)
    z2 = np.dot(a1, W2) + b2
    probs = softmax(z2)
    return np.argmax(probs, axis=1)

# Plot the decision boundary
#plot_decision_boundary(lambda x: predict(model, x))
#plt.title("Decision Boundary for hidden layer size 3")
#plt.show()

plt.figure(figsize=(16, 25))
hidden_layer_dimensions = [1, 2, 3, 4, 5, 20, 50]
for i, nn_hdim in enumerate(hidden_layer_dimensions):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer size %d' % nn_hdim)
    model_params = neural_network_model(nn_hdim)
    plot_decision_boundary(lambda x: predict(model_params, x))
plt.show()