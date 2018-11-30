import numpy as np

# (rows, cols)

# [f_a, f_b, f_c, f_d, f_e]
# [l_a, l_b]
#
# if f_b == 1 AND f_d == 1 
# l_a == 1
#
# if f_a == 0 and f_c == 1
# l_b == 1

# Each row in this array is a complete set of features for 1 trial
feature_data = np.array(
                    [
                        [1, 0, 0, 1, 0],
                        [0, 1, 0, 1, 1],
                        [0, 0, 1, 0, 1],
                        [0, 1, 0, 1, 1],
                        [1, 1, 1, 1, 1],
                        [0, 0, 1, 0, 0],
                        [0, 1, 1, 1, 0]
                    ])

# Each row in this array is a complete set of labels that corresponds to a complete set of features
label_data = np.array(
                    [
                        [0, 0],
                        [1, 0],
                        [0, 1],
                        [1, 0],
                        [1, 0],
                        [0, 1],
                        [1, 1]
                    ])

# Set up your hidden layer weights
# Every layer will have 1 more layer than your data set because of the bias
# I'll do a multi layer network, even though this problem is not complex, just to show you how
syn0 = np.random.random((5 + 1, 5))
syn1 = np.random.random((5 + 1, 3))
syn2 = np.random.random((3 + 1, 2))

def sigmoid(x, deriv=False):
    '''
    ::param x np.array:: value to apply sigmoid function to
    ::param deriv boolean:: flag to use derivative of sigmoid
    ::return np.array::

    The sigmoid activation function also called a "non-linearity function".

    Maps any value of x to a value between 0 and 1. This is useful to turn a number into a probability (0 - 100%).
    '''

    output = 1 / (1 + np.exp(-x))

    if deriv:
        return output * (1 - output)
    else:
        return output

def relu(x, deriv=False):
    '''
    ::param x np.array:: value to apply relu function to
    ::param deriv boolean:: flag to use derivative of relu
    ::return np.array::

    The relu (rectafied linear unit) activation function. In recent times this has become a lot more popular
    than the sigmoid activation function, especially within a deeply connected network. This is because sigmoid 
    functions have a problem with vanishing gradients. As the x value gets large, the value output of the sigmoid
    gets incredibly small. The constant gradient of the relu results in much faster learning. Additionally,
    the computational complexity of relu is much lower than sigmoid.
    '''

    if deriv:
        first = x[x > 0] = 1
        second = x[x < 0] = 0.01 # 0.1 instead of 0 to add leaky ReLu and avoid vanishing gradient
        return second
    else:
        return np.maximum(x, 0, x)


def add_bias(input_data):
    '''
    Add a bias to the layer by appending a 1 to the end of each row
    '''

    row_count = len(input_data)
    bias = np.ones((row_count, 1))

    input_with_bias = np.hstack((input_data, bias))
    return input_with_bias


def train(feature_data, label_data, epocs, learning_rate):
    global syn0
    global syn1
    global syn2

    final_activation = sigmoid
    other_activation = relu

    for i in range(epocs):

        #
        # Forward propogation
        #

        # Input layer
        layer_0 = feature_data
        layer_0_b = add_bias(layer_0)

        layer_1 = np.dot(layer_0_b, syn0)
        layer_1 = other_activation(layer_1)

        layer_1_b = add_bias(layer_1)
        layer_2 = np.dot(layer_1_b, syn1)
        layer_2 = other_activation(layer_2)

        layer_2_b = add_bias(layer_2)
        layer_3 = np.dot(layer_2_b, syn2)
        layer_3 = final_activation(layer_3)

        #
        # Backward propogation
        # get deltas

        # For all but first layer, dot the upstream error with current synapse transpose
        # Delta is always the error * deriv_actiation(layer)
        layer_3_error = label_data - layer_3
        layer_3_delta = layer_3_error * final_activation(layer_3, deriv=True)


        layer_2_error = np.dot(layer_3_error, syn2.T)
        layer_2_delta = layer_2_error * other_activation(layer_2_b, deriv=True)

        layer_2_error = layer_2_error[:,:-1] # Remove last column (bias)
        layer_1_error = np.dot(layer_2_error, syn1.T)
        layer_1_delta = layer_1_error * other_activation(layer_1_b, deriv=True)

        #
        # Backward propogation
        # update weights

        layer_2_delta = layer_2_delta[:,:-1] # to remove bias delta
        layer_1_delta = layer_1_delta[:,:-1] # to remove bias delta

        # Dot the transpose of the layer with the upstream delta
        syn2 += np.dot(layer_2_b.T, layer_3_delta) * learning_rate
        syn1 += np.dot(layer_1_b.T, layer_2_delta) * learning_rate
        syn0 += np.dot(layer_0_b.T, layer_1_delta) * learning_rate

        if i % 1000 == 0:
            print('Current error: {}'.format(str(np.sum(layer_3_error))))

    print('\nComplete\n')
    print('Real Data---v')
    print(label_data)
    print('Predicted final--v')
    print(layer_3)

train(feature_data, label_data, 8000, 0.05)