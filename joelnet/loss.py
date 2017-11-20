"""
We will train our models using loss functions
that indicate how good or bad our predictions are
for known input/output pairs. Then we can use the
gradients of this loss function with respect to the
various parameters of the net to adjust the parameters
and make our predictions better
"""
import numpy as np

from joelnet.tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError

class MSE(Loss):
    """
    This is actually total squared error
    not mean squared error
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)
