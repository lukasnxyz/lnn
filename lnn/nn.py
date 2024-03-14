import numpy as np
import random
from tensor import Tensor

class Neuron:
    def __init__(self, nin, actf=None):
        self.w = [Tensor(random.uniform(-1, 1)) for _ in range(nin)] # this may cause problems with orgate rn
        self.b = Tensor(random.uniform(-1, 1))
        self.actf = actf if actf is not None else Tensor.relu

    def __call__(self, x) -> Tensor:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b) # here too problems
        out = self.actf(act)
        return out

    def parameters(self) -> Tensor:
        return self.w + [self.b]

# A vertical layer of neurons
class Layer:
    def __init__(self, nin, nout, actf=None):
        self.neurons = [Neuron(nin, actf=actf) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

# Multi-Layer Perceptron (the network of hidden layers basically)
class MLP:
    def __init__(self, nin, nouts, actf=None): # list of nouts
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], actf=actf) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

# TESTING HERE
def main():
    from keras.datasets import mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    #X_train = Tensor(X_train)
    #X_test = Tensor(X_test)
    #Y_train = Tensor(Y_train)

    # how do I model this correctly? I'm not understanding
    n = MLP(784, [4, 10])

    epochs = 1
    batch_size = 5

    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            Y_batch = Y_train[i:i+batch_size]

            #ypred = [n(x) for x in X_batch]
            l = n(X_batch[0])

            print(l)
            #print(ypred)
            print(Y_batch)
            return

            '''
            ypred = []
            for i, x in enumerate(X_batch):
                nx = n(x)
                nxs = [item.data[0] for item in nx]
                ypred.append(nxs)
            '''

            # MSE Loss
            # loss has to be a single number here
            # and also make it so that it all stays a Tensor
            loss = sum((yact - yp)**2 for yp, yact in zip(Y_batch, ypred))
            print(loss)

            # backward pass
            for p in n.parameters():
                p.grad = np.zeros(p.data.shape)
            loss.backward()

            # update
            lr: np.float32 = 0.01 # this being a python float slowed everything down
            for p in n.parameters():
                p.data += -lr * p.grad # this is basically the optimizer

if __name__ == "__main__":
    main()
