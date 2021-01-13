import numpy as np
from enum import Enum
from activation_functions import *


class BwdPropType(Enum):
    Default = 0,
    L2 = 1,
    Backdrop = 2,


class BackwardPropagation:
    prop_type = BwdPropType.Default
    hyp_lambda = 0
    keep_prob = 1

    def __init__(self, prop_type=BwdPropType.Default, hyp_lambda=0.1, keep_prob=1):
        self.prop_type = prop_type
        self.hyp_lambda = hyp_lambda
        self.keep_prob = keep_prob

    def compute(self, al, y, caches):
        return self.__l_model_backward(al, y, caches)

    def __l_model_backward(self, al, y, caches):
        grads = {}
        length = len(caches)
        m = al.shape[1]
        y = y.reshape(al.shape)

        d_al = -(np.divide(y, al) - np.divide(1. - y, 1. - al))
        d_al[np.isnan(d_al)] = 0.

        current_cache = caches[length - 1]
        grads["dA" + str(length - 1)], grads["dW" + str(length)], grads[
            "db" + str(length)] = self.__linear_activation_backward(d_al,
                                                                    current_cache,
                                                                    ActivationFunction.Sigmoid)

        for i in reversed(range(length - 1)):
            current_cache = caches[i]
            da_prev_temp, dw_temp, db_temp = self.__linear_activation_backward(grads['dA' + str(i + 1)],
                                                                               current_cache,
                                                                               ActivationFunction.ReLU)
            grads["dA" + str(i)] = da_prev_temp
            grads["dW" + str(i+1)] = dw_temp
            grads["db" + str(i+1)] = db_temp

        return grads

    def __linear_activation_backward(self, d_a, cache, activation):
        linear_cache, activation_cache = cache

        if activation == ActivationFunction.ReLU:
            d_z = relu_backward(d_a, activation_cache)
        elif activation == ActivationFunction.Sigmoid:
            d_z = sigmoid_backward(d_a, activation_cache)
        else:
            raise Exception('Unknown activation function {}'.format(activation))

        if len(linear_cache) == 4:
            return self.__linear_backward_with_backdrop(d_z, linear_cache)
        else:
            return self.__linear_backward(d_z, linear_cache)

    def __linear_backward(self, d_z, cache):
        a_prev, w, b = cache
        m = a_prev.shape[1]

        if self.prop_type == BwdPropType.L2:
            d_w = (1 / m) * np.dot(d_z, a_prev.T) + (self.hyp_lambda/m)*w
        else:
            d_w = (1 / m) * np.dot(d_z, a_prev.T)

        d_b = (1 / m) * np.sum(d_z, axis=1, keepdims=True)
        d_a_prev = np.dot(w.T, d_z)

        assert (d_a_prev.shape == a_prev.shape)
        assert (d_w.shape == w.shape)
        assert (d_b.shape == b.shape)

        return d_a_prev, d_w, d_b

    def __linear_backward_with_backdrop(self, d_z, cache):
        a_prev, w, b, d = cache
        m = a_prev.shape[1]

        if self.prop_type == BwdPropType.L2:
            d_w = (1. / m) * np.dot(d_z, a_prev.T) + (self.hyp_lambda/m)*w
        else:
            d_w = (1. / m) * np.dot(d_z, a_prev.T)

        d_b = (1. / m) * np.sum(d_z, axis=1, keepdims=True)
        d_a_prev = np.dot(w.T, d_z)
        d_a_prev = (d_a_prev * d) / self.keep_prob

        assert (d_a_prev.shape == a_prev.shape)
        assert (d_w.shape == w.shape)
        assert (d_b.shape == b.shape)

        return d_a_prev, d_w, d_b


