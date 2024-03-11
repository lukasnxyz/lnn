import numpy as np
import random
from tensor import Tensor
from tqdm import trange

class Neuron:
    def __init__(self, nin):
        self.w = [Tensor(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Tensor(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
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

# TESTING HERE
def main():
    # OR gate
    X_train = Tensor([
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0]])

    Y_train = Tensor([
        [0.0],
        [1.0],
        [1.0],
        [1.0]])

    n = MLP(2, [4, 4, 1])

    epochs = 1000

    for epoch in (t := trange(epochs)):
        # forward pass
        ypred = [n(x) for i, x in enumerate(X_train.data)]

        loss = sum((yact - yp)**2 for yp, yact in zip(Y_train.data, ypred))
        t.set_description("loss: %.5f" % (loss.data[0]))

        # backward pass
        for p in n.parameters():
            p.grad = np.zeros(p.data.shape)
        loss.backward()

        # update
        lr: np.float32 = 0.01
        for p in n.parameters():
            p.data += -lr * p.grad # this is basically the optimizer

    for y in ypred:
        print(y)

    '''

    from keras.datasets import mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], -1)
    Y_train = Y_train.reshape(-1, 1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    X_train = Tensor(X_train)
    X_test = Tensor(X_test)
    Y_train = Tensor(Y_train)

    n = MLP(784, [4, 4, 10])

    epochs = 1
    batch_size = 5

    for epoch in range(epochs):
        for i in range(0, len(X_train.data), batch_size):
            X_batch = Tensor(X_train.data[i:i+batch_size])
            Y_batch = Tensor(Y_train.data[i:i+batch_size])

            # forward pass
            ypred = []
            for i, x in enumerate(X_batch.data):
                l = n(x)
                li = []
                for item in l:
                    num = item.data[0]
                    li.append(num)
                ypred.append(Tensor(np.argmax(li)))

            # MSE Loss
            loss = sum((yact - yp)**2 for yp, yact in zip(Y_batch.data, ypred)) # subtraction overflow here
            print(loss)

            # backward pass
            for p in n.parameters():
                p.grad = np.zeros(p.data.shape)
            loss.backward() # this isn't working because all the gradients are still 0 afterwards

            # update
            lr: np.float32 = 0.01 # this being a python float slowed everything down
            for p in n.parameters():
                p.data += -lr * p.grad # this is basically the optimizer
    '''

if __name__ == "__main__":
    main()
