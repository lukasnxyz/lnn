# This is a neural network model build based of off Andrej Karpathy's
# micrograd and his YouTube tutorial on it.
# Trying to create a multi-layer perceptron neural network to solve
# MNIST classification.

import numpy as np
import random

class Value: # Change this to Tensor
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, {self, other}, '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __radd__(self, other): # other + self
        return self + other

    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, {self, other}, '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other): # other * self (special Python function)
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def __truediv__(self, other): # self / other
        return self * other**-1

    def exp(self):
        x = self.data
        out = Value(np.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (np.exp(2*x) - 1)/(np.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

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

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x): # w * x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.relu()
        return out

    def parameters(self):
        return self.w + [self.b]

# A vertical layer of neurons
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

# Multi-Layer Perceptron (the network of hidden layers basically)
class MLP:
    def __init__(self, nin, nouts): # list of nouts
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

def main():
    from keras.datasets import mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    #X_train = X_train.reshape(X_train.shape[0], -1)
    #X_test = X_test.reshape(X_test.shape[0], -1)

    X_train = Value(X_train)
    X_test = Value(X_test)
    Y_train = Value(Y_train)

    n = MLP(784, [128, 128, 10])

    epochs = 1
    batch_size = 5

    for epoch in range(epochs):
        for i in range(0, len(X_train.data), batch_size):
            X_batch = X_train.data[i:i+batch_size]
            Y_batch = Y_train.data[i:i+batch_size]

            # forward pass
            ypred_i = [n(x) for i, x in enumerate(X_batch)] # this is where it is really slow (idk why)
                                                                               # speeds up when in shape (28, 28)

            ypred = []
            for y in ypred_i:
                print(y)
                ypred.append(np.argmax(y))

            ypred = np.array(ypred)

            print("ypred", len(ypred))
            print(type(ypred))
            print(ypred)
            print("Y_batch", Y_batch.shape)
            print(type(Y_batch))
            print(Y_batch)

            loss = sum((print(yact[0].data, yp), (yact - yp)**2) for yp, yact in zip(Y_batch, ypred)) # MSE Loss
            # yact is a value array
            print("loss:", loss)

            # backward pass
            for p in n.parameters():
                p.grad = 0.0
            loss.backward()

            # update
            lr = 0.001
            for p in n.parameters():
                p.data -= lr * p.grad # this is basically the optimizer

            print("loss: %.4f" % (loss.data))

if __name__ == "__main__":
    main()
