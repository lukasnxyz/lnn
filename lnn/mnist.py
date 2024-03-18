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

			r1 = n(X_batch[0])
			print(r1)
			return

			#ypred = [n(x) for x in X_batch]

			#print(ypred)
			#print(Y_batch)
			#return

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
