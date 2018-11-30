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
        if x > 0:
            return 1
        else:
            return 0
    else:
        return max(0, x)


def add_bias(input_data):

    pass


def train(feature_data, label_data, epocs, learning_rate):
    # syn0 = np.random.random((5 + 1, 5)) 6, 5
    # syn1 = np.random.random((5 + 1, 3)) 6, 3
    # syn2 = np.random.random((3 + 1, 2)) 4, 2

    for _ in range(epocs):

        layer_0 = feature_data
