import numpy as np
import primitives as p


class Conv1DLayer:
    def __init__(self, out_channels, kernel_size):
        raise NotImplementedError("Conv1DLayer.__init__ is not implemented")

    def forward(self, x):        
        raise NotImplementedError("Conv1DLayer.forward is not implemented")

    def parameters(self):
        raise NotImplementedError("Conv1DLayer.parameters is not implemented")


class MaxPool1DLayer:
    def __init__(self, pool_size, stride):
        raise NotImplementedError("MaxPool1DLayer.__init__ is not implemented")

    def forward(self, x):
        raise NotImplementedError("MaxPool1DLayer.forward is not implemented")

    def parameters(self):
        raise NotImplementedError("MaxPool1DLayer.parameters is not implemented")
