import numpy as np
import matplotlib.pyplot as plt

# comment the entire code, explain everything, and generate graphs
# from this code create a blog post on training a neural network from scratch
# Basic Neural Network from Scratch

def sigmoid(x):
	return 1/(1 + np.exp(-x))

def sigmoid_deriv(x):
	return sigmoid(x) + (1 - sigmoid(x))

def forward(x, w1, w2):
	h1 = x.dot(w1)
	a1 = sigmoid(h1)

	h2 = a1.dot(w2)
	a2 = sigmoid(h2)

	return a2

def generate_wt(x, y):
	l = []
	for i in range(x * y):
		l.append(np.random.randn())
	return np.array(l).reshape(x, y)

def loss(out, y):
	s = np.square(out - y)
	s = np.sum(s) / len(y)

	return s

def back_prop(x, y, w1, w2, alpha):
	h1 = x.dot(w1)
	a1 = sigmoid(h1)

	h2 = a1.dot(w2)
	a2 = sigmoid(h2)

	d2 = (a2 - y)
	d1 = np.multiply(w2.dot(d2.transpose()).transpose(), np.multiply(a1, 1 - a1))

	w1_adj = x.transpose().dot(d1)
	w2_adj = a1.transpose().dot(d2)

	w1 = w1 - (alpha * w1_adj)
	w2 = w2 - (alpha * w2_adj)

	return w1, w2

def train(x, y, w1, w2, alpha=0.01, epochs=1):
	losses = []

	for j in range(1, epochs+1):
		l = []

		for i in range(len(x)):
			out = forward(x[i], w1, w2)
			print(out)
			print(y[i])
			l.append(loss(out, y[i]))

			w1, w2 = back_prop(x[i], 	y[i], w1, w2, alpha)

		print("epoch:", j)
		losses.append(sum(l) / len(x))
	
	return losses, w1, w2

def main():
	x = [[0.0, 0.0],
			[0.0, 1.0],
			[1.0, 0.0],
			[1.0, 1.0]]
	
	y = [0.0, 1.0, 1.0, 0.0]

	x = np.array(x, dtype=np.float32)
	y = np.array(y, dtype=np.float32).reshape(-1, 1)

	w1 = generate_wt(2, 4)
	w2 = generate_wt(4, 1)

	loss, w1, w2 = train(x, y, w1, w2, 0.01, 100)

	print("loss:", loss)
	print("prediction:", forward(x, w1, w2))

if __name__ == "__main__":
	main()