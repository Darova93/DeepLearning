import numpy as np
from enum import Enum


class ActivationFunction:
    Sigmoid = 0,
    ReLU = 1,


def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    cache = z

    return a, cache


def relu(z):
    a = np.maximum(0, z)

    assert (a.shape == z.shape)

    cache = z
    return a, cache


def relu_backward(d_a, cache):
    z = cache
    d_z = np.array(d_a, copy=True)
    d_z[z <= 0] = 0

    assert (d_z.shape == z.shape)

    return d_z


def sigmoid_backward(d_a, cache):
    z = cache

    s = 1. / (1. + np.exp(-z))
    d_z = d_a * s * (1. - s)

    assert (d_z.shape == z.shape)

    return d_z
