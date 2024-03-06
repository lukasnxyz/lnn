from __future__ import annotations
import numpy as np

# Todo:
# - Implement function class as well for forward and backward pass functions
# - Tensor.transpose


class Tensor:
    def __init__(self, data, _children=()):
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data, dtype=np.float32)

        self.grad = np.zeros(self.data.shape, dtype=np.float32)

        self._prev = set(_children)
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(data: {self.data}, grad: {self.grad})"

    def __add__(self, other) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = np.add(self.data, other.data)
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
        out_data = self.data * other.data
        out = Tensor(out_data, {self, other})

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other) -> Tensor:
        return self * other

    def __matmul__(self, other) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = np.matmul(self.data, other.data)
        out = Tensor(out_data, {self, other})

        def _backward():
            self.grad += np.matmul(other.data, out.grad)
            other.grad += np.matmul(self.data, out.grad)

        out._backward = _backward

        return out

    def __rmatmul__(self, other) -> Tensor:
        return self @ other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, other):
        assert isinstance(other, (int, np.float32)), "only supporting int/np.float32 powers for now"
        out_data = self.data ** other
        out = Tensor(out_data, (self,))

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) @ out.grad
        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * other**-1

    def relu(self):
        x = self.data
        out_data = np.maximum(0, self.data)
        out = Tensor(out_data, (self, ))

        def _backward():
            self.grad += 0 if x < 0 else 1
        out._backward = _backward

        return out

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
