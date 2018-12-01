import numpy as np

# Data key:
#
# Randomly picked pattern in first 2 elements
# If second element is 1 or both first and second are 1: output[0] = 1
# [0, 1, 0, 0]
# [1, 0]
#
# [1, 1, 0, 0]
# [1, 0]
#
# XOR in last 2 elements
# [0, 0, 0, 0]
# [0, 0]
#
# [0, 0, 0, 1]
# [0, 1]
#
# [0, 0, 1, 1]
# [0, 0]
#
# [0, 0, 1, 0]
# [0, 1]

#
# Create input data
# Each row in this array is a complete set of features for 1 trial
#
feature_data = np.array(
                    [
                        [1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 1],
                        [0, 0, 0, 1, 0],
                        [1, 1, 0, 1, 1],
                        [0, 0, 0, 0, 0],
                        [1, 0, 0, 1, 0],
                        [0, 1, 0, 1, 0]
                    ])

#
# Create output data
# Each row in this array is a complete set of labels that corresponds to a complete set of features
#
label_data = np.array(
                    [
                        [0, 0],
                        [1, 1],
                        [0, 1],
                        [1, 0],
                        [0, 0],
                        [0, 1],
                        [1, 1]
                    ])

#
# Set up your hidden layer weights to be random value in range -1, 1
# Every layer will have 1 more row than the previous layer because we will be adding a bias
#
syn0 = 2 * np.random.random((5 + 1, 5)) - 1
syn1 = 2 * np.random.random((5 + 1, 3)) - 1
syn2 = 2 * np.random.random((3 + 1, 2)) - 1


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

    The leaky relu (rectafied linear unit) activation function. In recent times this has become a lot more popular
    than the sigmoid activation function, especially within a deeply connected network. This is because sigmoid 
    functions have a problem with vanishing gradients. As the x value gets large, the value output of the sigmoid
    gets incredibly small. The constant gradient of the relu results in much faster learning. Additionally,
    the computational complexity of relu is much lower than sigmoid.
    '''

    if deriv:
        x[x > 0] = 1
        x[x < 0] = 0.01   #0.01 * x[x < 0]  # 0.1 instead of 0 to add leaky ReLu and avoid vanishing gradient
        return x
    else:
        return  np.maximum(x, 0.01 * x, x) #np.maximum(x, 0, x)


def add_bias(input_data):
    '''
    Add a bias to the layer by appending a 1 to the end of each row
    '''

    row_count = len(input_data)
    bias = np.ones((row_count, 1))

    input_with_bias = np.hstack((input_data, bias))
    return input_with_bias


def predict(weights, feature_data, final_activation, other_activation):
    '''
    ::param weights np.array::
    ::param test_data np.arry::
    ::return np.array::

    One forward pass of the data against weights to predit label
    '''
    syn0, syn1, syn2 = weights

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

    return layer_0_b, layer_1_b, layer_2_b, layer_3


def train(weights, feature_data, label_data, epocs, learning_rate):
    final_activation = sigmoid
    other_activation = relu
    syn0, syn1, syn2 = weights

    for i in range(epocs):

        #
        # Forward propogation
        #

        layer_0_b, layer_1_b, layer_2_b, layer_3 = predict(
                                                            weights, 
                                                            feature_data, 
                                                            final_activation, 
                                                            other_activation
                                                            )

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
        elif i % 5000 == 0 and i != 0:
            learning_rate *= 0.7

    print('\nComplete\n')
    print(label_data)
    layer_3[layer_3 > 0.5] = 1
    layer_3[layer_3 < 0.5] = 0
    print(layer_3)

train((syn0, syn1, syn2), feature_data, label_data, 20000, 0.05)


#
# NN has not been trained on any of the below test data
#
print('\n')
# Should be (1, 0)
test_data = np.array([[0, 1, 0, 0, 0]])
*a, guess = predict((syn0, syn1, syn2), test_data, sigmoid, relu)
print(guess)

# Should be (1, 0)
test_data = np.array([[1, 1, 0, 0, 0]])
*a, guess = predict((syn0, syn1, syn2), test_data, sigmoid, relu)
print(guess)

# Should be (0, 1)
test_data = np.array([[0, 0, 0, 1, 0]])
*a, guess = predict((syn0, syn1, syn2), test_data, sigmoid, relu)
print(guess)