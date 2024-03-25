from tensor import Tensor
import numpy as np

def layer_init(m, h):
  ret = np.random.uniform(-1., 1., size=(m,h))/np.sqrt(m*h)
  return ret.astype(np.float32)

class NET:
	def __init__(self):
		self.l1 = Tensor(layer_init(784, 128))
		self.l2 = Tensor(layer_init(128, 10))

	def forward(self, x):
		return x.mul(self.l1).relu().mul(self.l2)

def main():
	from keras.datasets import mnist
	(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

	X_train = X_train.reshape(X_train.shape[0], -1)
	X_test = X_test.reshape(X_test.shape[0], -1)

	model = NET()

	epochs = 1
	batch_size = 5

	for epoch in range(epochs):
		for i in range(0, len(X_train), batch_size):
			X_batch = Tensor(X_train[i:i+batch_size])
			Y_batch = Tensor(Y_train[i:i+batch_size])

			out = model.forward(X_batch)

			loss = out.mul(Y_batch).mean()
			print(loss)
			return
			loss.backward()

			# update
			lr: np.float32 = 0.01 # this being a python float slowed everything down
			for p in n.parameters():
				p.data += -lr * p.grad # this is basically the optimizer

if __name__ == "__main__":
	main()
