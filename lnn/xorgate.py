from nn import Layer
from tensor import Tensor
import numpy as np
from tqdm import trange

class XOR:
	def __init__(self):
		self.h1 = Layer(2, 4)
		self.act1 = Layer.actf
		self.output = Layer(4, 1)
		self.act_output = Layer.actf

	def forward(self, X):
		X = self.h1(X)
		X = self.act1(X, actf=Tensor.relu)
		X = self.output(X)
		exit()
		X = self.act_ouput(X, actf=Tensor.relu)
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
		#print(out)
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
