import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v3 import *


def initialize_parameters_deep(layer_dims):
    parameters = {}
    for i in range(1, len(layer_dims)):
        parameters['W'+str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) / np.sqrt(layer_dims[i-1])
        parameters['b'+str(i)] = np.zeros((layer_dims[i], 1))

        assert (parameters['W'+str(i)].shape == (layer_dims[i], layer_dims[i - 1]))
        assert (parameters['b'+str(i)].shape == (layer_dims[i], 1))

    return parameters


def linear_forward(a, w, b):
    z = np.dot(w, a) + b

    assert(z.shape == (w.shape[0], a.shape[1]))
    cache = (a, w, b)

    return z, cache


def linear_activation_forward(a_prev, w, b, activation):
    if activation == 'sigmoid':
        z, linear_cache = linear_forward(a_prev, w, b)
        a, activation_cache = sigmoid(z)
    elif activation == 'relu':
        z, linear_cache = linear_forward(a_prev, w, b)
        a, activation_cache = relu(z)
    else:
        raise Exception('Unknown activation function {}'.format(activation))

    assert (a.shape == (w.shape[0], a_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return a, cache


def l_model_forward(x, parameters):
    caches = []
    a = x
    number_of_layers = len(parameters)//2

    for layer_index in range(1, number_of_layers):
        a_prev = a
        a, cache = linear_activation_forward(
                    a_prev,
                    parameters['W'+str(layer_index)],
                    parameters['b'+str(layer_index)],
                    'relu')
        caches.append(cache)

    al, cache = linear_activation_forward(
                a,
                parameters['W'+str(number_of_layers)],
                parameters['b'+str(number_of_layers)],
                'sigmoid')
    caches.append(cache)

    assert (al.shape == (1, x.shape[1]))

    return al, caches


def compute_cost(a, y):
    m = y.shape[1]
    cost = -(1./m) * np.sum(y*np.log(a)+(1-y)*np.log(1-a), axis=1, keepdims=True)

    cost = np.squeeze(cost)
    assert (cost.shape == ())

    return cost


def linear_backward(d_z, cache):
    a_prev, w, b = cache
    m = a_prev.shape[1]

    d_w = (1/m) * np.dot(d_z, a_prev.T)
    d_b = (1/m) * np.sum(d_z, axis=1, keepdims=True)
    d_a_prev = np.dot(w.T, d_z)

    assert (d_a_prev.shape == a_prev.shape)
    assert (d_w.shape == w.shape)
    assert (d_b.shape == b.shape)

    return d_a_prev, d_w, d_b


def linear_activation_backward(d_a, cache, activation):
    linear_cache, activation_cache = cache

    if activation == 'relu':
        d_z = relu_backward(d_a, activation_cache)
        d_a_prev, d_w, d_b = linear_backward(d_z, linear_cache)
    elif activation == 'sigmoid':
        d_z = sigmoid_backward(d_a, activation_cache)
        d_a_prev, d_w, d_b = linear_backward(d_z, linear_cache)
    else:
        raise Exception('Unknown activation function {}'.format(activation))

    return d_a_prev, d_w, d_b


def l_model_backward(al, y, caches):
    grads = {}
    length = len(caches)
    m = al.shape[1]
    y = y.reshape(al.shape)

    d_al = -(np.divide(y, al) - np.divide(1-y, 1-al))

    current_cache = caches[length-1]
    grads["dA" + str(length-1)], grads["dW" + str(length)], grads["db" + str(length)] = linear_activation_backward(
                                                                                            d_al,
                                                                                            current_cache,
                                                                                            'sigmoid')

    for i in reversed(range(length - 1)):
        current_cache = caches[i]
        da_prev_temp, dw_temp, db_temp = linear_activation_backward(grads['dA' + str(i+1)], current_cache, 'relu')
        grads["dA" + str(i)] = da_prev_temp
        grads["dW" + str(i+1)] = dw_temp
        grads["db" + str(i+1)] = db_temp

    return grads


def update_parameter(initial_parameters, grads, learning_rate):
    total_layers = len(initial_parameters) // 2

    for i in range(total_layers):
        initial_parameters["W"+str(i+1)] = initial_parameters["W"+str(i+1)] - learning_rate*grads["dW"+str(i+1)]
        initial_parameters["b"+str(i+1)] = initial_parameters["b"+str(i+1)] - learning_rate*grads["db"+str(i+1)]

    return initial_parameters


def display_image(index):
    train_x_orig, train_y_data, test_x_orig, test_y_data, classes_data = load_data()
    plt.imshow(train_x_orig[index])
    plt.show()
    print("y = "+str(train_y_data[0, index])+". It's a " + classes[train_y_data[0, index]].decode("utf-8")+" picture.")


def nn_model(x, y, layers_dimensions, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    costs = []
    parameters = initialize_parameters_deep(layers_dimensions)

    for i in range(0, num_iterations):
        al, caches = l_model_forward(x, parameters)
        cost = compute_cost(al, y)
        grads = l_model_backward(al, y, caches)
        parameters = update_parameter(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost.squeeze()))
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


def predict(x, y, parameters):
    m = x.shape[1]
    n = len(parameters) // 2
    p = np.zeros((1, m))

    probas, caches = l_model_forward(x, parameters)

    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    # print results
    # print ("predictions: " + str(p))
    # print ("true labels: " + str(y))
    print("Accuracy: " + str(np.sum((p == y) / m)))

    return p


def print_mislabeled_images(set_classes, x, y, p):
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0)
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(x[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title(
            "Prediction: " +
            set_classes[int(p[0, index])].decode("utf-8")+" \n Class: " +
            set_classes[y[0, index]].decode("utf-8"))
    plt.show()


def evaluate_image(image_name, parameters, is_cat=True):
    image = np.array(Image.open(image_name).resize((64, 64))) / 255.
    flattened_image = image.reshape(image.shape[0]*image.shape[1]*image.shape[2], -1)
    my_predicted_image = predict(flattened_image, [is_cat*1.], parameters)

    # plt.imshow(image)
    # plt.show()
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[
        int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")


if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = (5.0, 4.0)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    train_x, train_y, test_x, test_y, classes = load_data()
    layers_dims = (12288, 20, 7, 5, 1)

    trained_parameters = nn_model(train_x,
                                  train_y,
                                  layers_dims,
                                  learning_rate=0.0075,
                                  num_iterations=2500,
                                  print_cost=True)

    predictions_train = predict(train_x, train_y, trained_parameters)
    predictions_test = predict(test_x, test_y, trained_parameters)

    evaluate_image('datasets/cat.png', trained_parameters)
