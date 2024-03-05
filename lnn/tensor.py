import numpy as np

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

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = np.add(self.data, other.data)
        out = Tensor(out_data, {self, other})

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = np.matmul(self.data, other.data)
        out = Tensor(out_data, {self, other})

        def _backward():
            self.grad += np.matmul(other.data, out.grad)
            other.grad += np.matmul(self.data, out.grad)
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return np.matmul(self, other)








    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def __pow__(self, other):
        assert isinstance(other, (int, np.float32)), "only supporting int/np.float32 powers for now"
        out = Value(self.data**other, (self,))

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def __truediv__(self, other): # self / other
        return self * other**-1

    def relu(self):
        x = self.data
        relu = np.maximum(0, self.data)
        out = Value(relu, (self, ))

        def _backward():
            self.grad += 0 if x < 0 else 1 # found online
        out._backward = _backward

        return out

    # One problem with this is if we reuse variables, then their gradients are stored and reused for
    # different equations
    # This is solved in the _backward() functions of __add__, __mul__, and tanh by accumulating the
    # gradients with += instead of just resetting them with = every time
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

def main():
    mat1 = Tensor([[1.3, 2.1, 3.8], [4.3, 5.9, 6.2]])
    mat2 = Tensor([[2.3, 3.1], [9.8, 1.3], [6.9, 7.2]])
    mat3 = mat1 * mat2
    print(mat1)
    print(mat2)
    print(mat3)

if __name__ == "__main__":
    main()
