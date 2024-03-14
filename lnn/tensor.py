from __future__ import annotations
import math
import numpy as np

#class Function:
#   def forward(self):
#   def backward(self):

class Tensor:
    def __init__(self, data, _children=()):
        self.data = np.array(data, dtype=np.float32)
        #if self.data.shape == ():
            #self.data = self.data.reshape((1,))

        self.grad = np.zeros(self.data.shape, dtype=np.float32)

        self._prev = set(_children)
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(data: {self.data}, grad: {self.grad})"

    def __add__(self, other) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data + other.data
        out = Tensor(out_data, {self, other})

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __radd__(self, other) -> Tensor:
        return self + other

    def __mul__(self, other) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = np.matmul(self.data, other.data)
        out = Tensor(out_data, {self, other})

        def _backward():
            self.grad += np.matmul(other.data, out.grad)
            other.grad += np.matmul(self.data, out.grad)
        out._backward = _backward

        return out

    def __rmul__(self, other) -> Tensor:
        return self * other

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, other):
        assert isinstance(other, (int, np.float32)), "only supporting int/np.float32 powers for now"
        out_data = self.data ** other
        out = Tensor(out_data, (self,))

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * other**-1

    def tanh(data):
        x = data.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Tensor(t, (data, ))

        def _backward():
            data.grad += (1.0 - t**2.0) * out.grad
        out._backward = _backward

        return out


    @staticmethod
    def relu(data): # doesn't work
        x = data.data
        out_data = np.maximum(0, x)
        out = Tensor(out_data, (data, ))

        def _backward():
            self.grad += (0 if x < 0 else 1) * out.grad
        out._backward = _backward

        return out

    #def sigmoid(self):

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()
