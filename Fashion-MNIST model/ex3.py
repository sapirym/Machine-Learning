import numpy as np


# LAYERS
def relu(vec):
    """
    Relu calculation in accordance to what was learned in class.
    :param vec: input for relu layer
    :return: output
    """
    return np.maximum(vec, 0)

def softmax(z2):
    """
    Softmax function calculation.
    :return: output
    """
    denominator = np.sum(np.exp(z2), axis=0)
    return np.exp(z2) / denominator

# WEIGHTS UPDATE
def update_weights(W1, W2, b1, b2, W1_grad, W2_grad, b1_grad, b2_grad):
    """
    Updates the weights and biases using the learning rate we chose, for each iteration.
    :return: updated weights
    """
    eta = 0.02
    W1 = W1 - (eta * W1_grad)
    W2 = W2 - (eta * W2_grad)
    b1 = b1 - (eta * b1_grad)
    b2 = b2 - (eta * b2_grad)
    return W1, W2, b1, b2


# FORWARD PROPAGATION.
def forward_propagation(W1, b1, W2, b2, x):
    """
     Call the Activation function ReLU and normalize
    :return: updated results
    """
    x_temp = np.reshape(x, (-1, 1))
    z1 = np.dot(W1, x_temp) + b1
    h1 = relu(z1)
    # Normalize
    if h1.max() != 0:
        h1 /= h1.max()
    z2 = np.dot(W2, h1) + b2
    y_hat = softmax(z2)
    return y_hat, z1, z2, h1


# BACK PROPAGATION
def backward_propagation(y_hat, z1, h1, W2, x, y):
    """
    Backwards Propagation function.
    :return: gradients.
    """
    h2 = y_hat
    correct_answer = int(y)
    # Derivative of H2
    h2[correct_answer] = h2[correct_answer] - 1
    b2_gradient = h2
    # Calculate the ReLU Derivative
    z1[z1 <= 0] = 0
    z1[z1 > 0] = 1
    # Calculate the necessary gradients.
    b1_gradient = np.dot(np.transpose(W2), h2) * z1
    w1_gradient = np.dot(b1_gradient, x)
    w2_gradient = np.dot(h2, np.transpose(h1))
    return w1_gradient, w2_gradient, b1_gradient, b2_gradient


def train_func(W1, b1, W2, b2, epochs, train_x, train_y):
    """
    Trains the algorithm and returns updated weights when done.
    :return: updated weights
    """
    # Normalize the training set to be less than 1.
    train_x /= 255
    for i in range(epochs):
        arr_zipped = list(zip(train_x, train_y))
        # Shuffle
        np.random.shuffle(arr_zipped)
        train_x, train_y = zip(*arr_zipped)

        # For each example and its respective value
        for x, y in zip(train_x, train_y):
            x = np.reshape(x, (1, 784))
            y_hat, z1, z2, h1 = forward_propagation(W1, b1, W2, b2, x)
            W1_grad, W2_grad, b1_grad, b2_grad = backward_propagation(y_hat, z1, h1, W2, x, y)
            W1, W2, b1, b2 = update_weights(W1, W2, b1, b2, W1_grad, W2_grad, b1_grad, b2_grad)
    return W1, W2, b1, b2


if __name__ == "__main__":
    """
    Main function
    """

    # learning parameters
    epochs = 20
    max = 1
    min = -1
    hidden_layers = 150

    # Initialize the weights and biases.
    W1 = np.random.uniform(min, max, [hidden_layers, 784])
    W2 = np.random.uniform(min, max, [10, hidden_layers])
    b1 = np.random.uniform(min, max, [hidden_layers, 1])
    b2 = np.random.uniform(min, max, [10, 1])


    x_training_set = np.loadtxt("train_x")
    y_training_set = np.loadtxt("train_y")
    x_testing_set = np.loadtxt("test_x")
    W1_trained, W2_trained, b1_trained, b2_trained = train_func(W1, b1, W2, b2, epochs, x_training_set,
                                                                y_training_set, )
