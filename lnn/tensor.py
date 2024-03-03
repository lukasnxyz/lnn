import numpy as np

class Tensor:
    def __init__(self, data: Union[np.ndarray, float], _children=()):
        self.data = data
        self._prev = set(_children)
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(data: {self.data}, grad: {self.grad})"

    '''
    Ops
        add
        mul
        sub
        div
        log
        pow
        relu
        tanh
        sigmoid
        exp

    '''
