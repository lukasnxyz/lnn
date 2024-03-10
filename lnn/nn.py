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

# TESTING HERE
def main():
    from keras.datasets import mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    X_train = Tensor(X_train)
    X_test = Tensor(X_test)
    Y_train = Tensor(Y_train)

    n = MLP(1, [2, 1])

    epochs = 2
    batch_size = 32

    for epoch in range(epochs):
        for i in (t := trange(0, len(X_train.data), batch_size)):
            X_batch = X_train.data[i:i+batch_size]
            Y_batch = Y_train.data[i:i+batch_size]

            # forward pass
            ypred_i = [n(x) for i, x in enumerate(X_batch)]

            ypred = []
            for y in ypred_i:
                ypred.append(Tensor(np.argmax(y)))

            # MSE Loss
            #loss = sum((print(type(yact.data), type(yp)), (yact - yp)**2) for yp, yact in zip(Y_batch, ypred)) # subtraction overflow here
            loss = sum((yact - yp)**2 for yp, yact in zip(Y_batch, ypred)) # subtraction overflow here
            t.set_description("loss: %.3f" % (loss.data))

            # backward pass
            for p in n.parameters():
                p.grad = 0.0
            loss.backward() # this isn't working because all the gradients are still 0 afterwards

            #for p in n.parameters():
                #print(p.grad)

            # update
            lr: np.float32 = 0.001 # this being a python float slowed everything down
            for p in n.parameters():
                p.data += -lr * p.grad # this is basically the optimizer

if __name__ == "__main__":
    main()
