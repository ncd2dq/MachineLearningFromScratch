import numpy as np
import matplotlib.pyplot as plt
from math import floor
from functools import partial
from typing import List, Union, Tuple


def find_offsets(minimum: int, maximum: int) -> Tuple[int]:
    '''
    Finds the required coefficient and offset to map a random number generator that
    produces between (0, 1) to a given minimum/maximum range
    '''
    # Find range
    coefficient_range = maximum - minimum

    # Find offset
    offset = abs(maximum - coefficient_range)
    if maximum < coefficient_range:
        offset *= -1

    return coefficient_range, offset


def random_polynomial_coefficients(minimum: int, maximum: int, degree: int) -> List[float]:
    '''
    Create a polynomial of form:
    ax^n + bx^n-1 + cx^n-2 .... + yx^1 + z

    ::param min:: Minimum coefficient value
    ::param max:: Maximum coefficient value

    ::return:: coefficients should be interpreted as in ascending order for simplicity
    '''
    if maximum <= minimum:
        raise ValueError('Maximum must be larger than minimum')

    coefficient_range, offset = find_offsets(minimum, maximum)
    coefficients = np.random.random((degree + 1)) * coefficient_range + offset

    return coefficients


def apply_polynomial(coefficients: List[float], x: Union[float, int]) -> Union[float, int]:
    '''
    Determine the return value of a polynomial with given coefficients at value x
    '''
    y = sum(coefficients[i] * (x**i) for i in range(len(coefficients)))

    return y


def create_data(minimum: int, maximum: int, 
                coefficients: List[float], percent_training_data: float =0.5, 
                data_point_count: int=10000) -> Tuple[List[float]]:
    '''
    Creates a sample set of data for testing and training based on the polynomial 
    with given coefficients.
    
    ::params minimum, maximum:: the min/max values on the x domain to evaluate 
    the polynomial
    ::return train_x, train_y, test_x, test_y:: -> ([x_data], [y_data], [x_data], [y_data])
    '''

    if percent_training_data <= 0:
        raise ValueError('Percent Training Data Must Be Positive')
    elif percent_training_data >= 1:
        raise ValueError('Percent Training Data Must Be Less Than 1')
    elif maximum <= minimum:
        raise ValueError('Maximum Must Be Larger Than Minimum')

    apply_poly_partial = partial(apply_polynomial, coefficients)
    training_data_count = floor(data_point_count * percent_training_data)
    coefficient_range, offset = find_offsets(maximum, minimum)

    x_points = np.random.random((data_point_count)) * coefficient_range + offset
    y_points = np.array(list(map(apply_poly_partial, x_points)))

    training_x_points = x_points[:training_data_count]
    training_y_points = y_points[:training_data_count]
    testing_x_points = x_points[training_data_count:]
    testing_y_points = y_points[training_data_count:]

    return training_x_points, training_y_points, testing_x_points, testing_y_points


def plot_data(data: List[List[float]]):
    '''
    Graph the training and testing data

    ::param data:: training/testing x and y arrays
    '''
    plt.scatter(data[0], data[1], color='blue', s=2, label='training')
    plt.scatter(data[2], data[3], color='red', s=2, label='testing')
    plt.legend()
    plt.show()


coefs = random_polynomial_coefficients(-1, 1, 4)
data = create_data(-6, 6, coefs, 0.5, data_point_count=100)
plot_data(data)


