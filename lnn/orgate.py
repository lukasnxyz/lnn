from nn import MLP
from tensor import Tensor
import numpy as np
from tqdm import trange

# Works if you change activation function to tanh in nn.py

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

    epochs = 2000

    for epoch in (t := trange(epochs)):
        # forward pass
        # this should be a tensor of a 2d array batch_size x 10
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

if __name__ == "__main__":
    main()
