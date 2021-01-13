import numpy as np
from enum import Enum


class CostType(Enum):
    Default = 0
    L2 = 1


class Cost:
    cost_type = CostType.Default
    hyp_lambda = 0

    def __init__(self, cost_type=CostType.Default, hyp_lambda=0.1):
        self.cost_type = cost_type
        self.hyp_lambda = hyp_lambda
        # np.seterr('raise')

    def compute(self, a, y, weights=[]):
        cost = 0
        if self.cost_type == CostType.Default:
            cost = self.__cross_entropy_cost(a, y)
        elif self.cost_type == CostType.L2:
            cost = self.__l2_regularization_cost(a, y, weights)
        else:
            raise Exception('Invalid cost type')

        assert (cost.shape == ())

        return cost

    def __l2_regularization_cost(self, a, y, parameters):
        m = y.shape[1]
        weight_sum = 0

        weights = filter(lambda x: x.startswith('W'), parameters)
        for item in weights:
            weight_sum = weight_sum + np.sum(parameters[item], keepdims=True)

        cross_entropy_cost = self.__cross_entropy_cost(a, y)
        l2_regularization_cost = (self.hyp_lambda / (2 * m)) * weight_sum

        return np.squeeze(cross_entropy_cost + l2_regularization_cost)

    @staticmethod
    def __cross_entropy_cost(a, y):
        m = y.shape[1]
        logs = np.multiply(-np.log(a), y) + np.multiply(-np.log(1-a), 1-y)
        cost = 1./m * np.nansum(logs, axis=1, keepdims=True)

        return np.squeeze(cost)
