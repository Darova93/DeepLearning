import numpy as np
from enum import Enum
from activation_functions import *


class FwdPropType(Enum):
    Default = 0,
    Backdrop = 1,


class ForwardPropagation:
    prop_type = FwdPropType.Default
    keep_prob = 1

    def __init__(self, prop_type=FwdPropType.Default, keep_prob=1):
        self.prop_type = prop_type
        self.keep_prob = keep_prob

    def compute(self, x, parameters):
        if self.prop_type == FwdPropType.Backdrop:
            return self.__l_model_forward_with_backdrop(x, parameters)
        else:
            return self.__l_model_forward(x, parameters)

    def __l_model_forward(self, x, parameters):
        caches = []
        a = x
        number_of_layers = len(parameters) // 2

        for layer_index in range(1, number_of_layers):
            a_prev = a
            a, cache = self.__linear_activation_forward(
                a_prev,
                parameters['W' + str(layer_index)],
                parameters['b' + str(layer_index)],
                ActivationFunction.ReLU)
            caches.append(cache)

        al, cache = self.__linear_activation_forward(
            a,
            parameters['W' + str(number_of_layers)],
            parameters['b' + str(number_of_layers)],
            ActivationFunction.Sigmoid)
        caches.append(cache)

        assert (al.shape == (1, x.shape[1]))

        return al, caches

    def __l_model_forward_with_backdrop(self, x, parameters):
        caches = []
        a = x
        number_of_layers = len(parameters) // 2
        current_d = None
        np.random.seed(1)
        for layer_index in range(1, number_of_layers):
            a_prev = a
            a, cache, current_d = self.__linear_activation_forward_with_backdrop(
                a_prev,
                parameters['W' + str(layer_index)],
                parameters['b' + str(layer_index)],
                ActivationFunction.ReLU,
                current_d)
            caches.append(cache)

        al, cache = self.__linear_activation_forward(
            a,
            parameters['W' + str(number_of_layers)],
            parameters['b' + str(number_of_layers)],
            ActivationFunction.Sigmoid,
            current_d)
        caches.append(cache)

        assert (al.shape == (1, x.shape[1]))

        return al, caches

    def __linear_activation_forward(self, a_prev, w, b, activation, current_d=None):
        z, linear_cache = self.__linear_forward(a_prev, w, b)
        if activation == ActivationFunction.Sigmoid:
            a, activation_cache = sigmoid(z)
        elif activation == ActivationFunction.ReLU:
            a, activation_cache = relu(z)
        else:
            raise Exception('Unknown activation function {}'.format(activation))

        if current_d is not None:
            p1, p2, p3 = linear_cache
            linear_cache = (p1, p2, p3, current_d)

        assert (a.shape == (w.shape[0], a_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return a, cache

    def __linear_activation_forward_with_backdrop(self, a_prev, w, b, activation, current_d=None):
        z, linear_cache = self.__linear_forward(a_prev, w, b)
        if activation == ActivationFunction.Sigmoid:
            a, activation_cache = sigmoid(z)
        elif activation == ActivationFunction.ReLU:
            a, activation_cache = relu(z)
        else:
            raise Exception('Unknown activation function {}'.format(activation))

        d = np.random.rand(a.shape[0], a.shape[1])
        d = (d < self.keep_prob).astype(int)
        a = (a * d) / self.keep_prob
        if current_d is not None:
            p1, p2, p3 = linear_cache
            linear_cache = (p1, p2, p3, current_d)

        assert (a.shape == (w.shape[0], a_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return a, cache, d

    @staticmethod
    def __linear_forward(a, w, b):
        z = np.dot(w, a) + b

        assert (z.shape == (w.shape[0], a.shape[1]))
        cache = (a, w, b)

        return z, cache