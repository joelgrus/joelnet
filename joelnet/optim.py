"""
An optimizer uses the computed gradients
to adjust the parameters of a neural net
"""

from joelnet.nn import NeuralNet
from joelnet.tensor import Tensor


class Optimizer:
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr: float) -> None:
        self.lr = lr

    def step(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad
