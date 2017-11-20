"""
The canonical example of a function that can't
be learned by a linear layer alone is XOR.
"""
import numpy as np

from joelnet.nn import NeuralNet
from joelnet.layers import Linear, Tanh
from joelnet.train import train

inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

net = NeuralNet([
    Linear(input_size=2, output_size=2),
    Tanh(),
    Linear(input_size=2, output_size=2)
])

train(net, inputs, targets, num_epochs=5000)

for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)
