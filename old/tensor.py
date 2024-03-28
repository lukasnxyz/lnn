from functools import partialmethod
import numpy as np

class Tensor:
	def __init__(self, data, _children=()):
		assert(isinstance(data, np.ndarray))
		self.data = data
		self.grad = np.zeros_like(data)
		self._prev = set(_children)
		self._backward = lambda: None

	def __repr__(self):
		return f"Tensor(data: {self.data}, grad: {self.grad})"

	def add(self, y):
		out = self.data + y.data
		ret = Tensor(out, {self, y})

		def _backward():
			self.grad += 1.0 * ret.grad
			y.grad += 1.0 * ret.grad
		ret._backward = _backward

		return ret

	def mul(self, y):
		#out = np.dot(self.data, y.data)
		out = self.data * y.data
		ret = Tensor(out, {self, y})

		def _backward():
			self.grad += y.data * ret.grad
			y.grad += self.data * ret.grad
		ret._backward = _backward

		return ret

	def relu(self):
		out =	np.maximum(0, self.data)
		ret = Tensor(out, {self})

		def _backward():
			self.grad += (1 - t**2) * out.grad
		ret._backward = _backward

		return ret

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

def main():
	a = Tensor(np.eye(2))
	b = Tensor(np.array([[1.0, 4.0], [3.0, 7.0]], dtype=np.float32))

	c = a.add(b)
	d = c.mul(c)

	d.backward()
	print("a:", a)
	print("b:", b)
	print("c:", c)
	print("d:", d)

if __name__ == "__main__":
	main()
