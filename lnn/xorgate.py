from nn import Layer
from tensor import Tensor
import numpy as np
from tqdm import trange

# Works if you change activation function to tanh in nn.py

class XOR:
	def __init__(self):
		self.h1 = Layer(2, 2)
		self.output = Layer(2, 1)

	def forward(self, X):
		X = self.output(self.h1(X))

		return X

def main():
	X_train = Tensor([
		[0.0, 0.0],
		[0.0, 1.0],
		[1.0, 0.0],
		[1.0, 1.0]])

	Y_train = Tensor([
		[0.0],
		[1.0],
		[1.0],
		[0.0]])

	model = XOR()
	epochs = 200

	for epoch in (t := trange(epochs)):
		out = model.forward(X_train)
		print(out)
		return

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
