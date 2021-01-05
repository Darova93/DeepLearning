import numpy as np
from enum import Enum


class InitType(Enum):
    Default = 0
    Xavier = 1
    He = 2


class ParameterInitializer:
    layer_dims = []
    init_type = InitType.Default
    factors = {
        InitType.Default: lambda x: 1./np.sqrt(x),
        InitType.Xavier: lambda x: np.sqrt(1./x),
        InitType.He: lambda x: np.sqrt(2./x),
    }

    def __init__(self, layer_dims, init_type=InitType.Default):
        self.layer_dims = layer_dims
        self.init_type = init_type
        np.random.seed(1)

    def initialize(self):
        return self.__create_parameters(self.factors[self.init_type])

    def __create_parameters(self, factor):
        parameters = {}
        for i in range(1, len(self.layer_dims)):
            parameters['W'+str(i)] = np.random.randn(self.layer_dims[i], self.layer_dims[i-1]) \
                                          * factor(self.layer_dims[i-1])
            parameters['b'+str(i)] = np.zeros((self.layer_dims[i], 1))

        self.__consistency_check(parameters)
        return parameters

    def __consistency_check(self, parameters):
        for i in range(1, len(self.layer_dims)):
            assert (parameters['W' + str(i)].shape == (self.layer_dims[i], self.layer_dims[i - 1]))
            assert (parameters['b' + str(i)].shape == (self.layer_dims[i], 1))
