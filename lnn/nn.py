import numpy as np
import random
from tensor import Tensor

class Neuron:
	def __init__(self, nin):
		self.w = Tensor([random.uniform(-1, 1) for _ in range(nin)])
		self.b = Tensor(random.uniform(-1, 1))

	def __call__(self, x: Tensor) -> Tensor:
		out = self.w * x.data.transpose() + self.b
		return out

	def __repr__(self):
		return f"Neuron: (w: {self.w.data.shape} b: {self.b.data.shape})"

	def parameters(self) -> Tensor:
		return self.w + self.b

# A vertical layer of neurons
class Layer:
	def __init__(self, nin, nout):
		self.neurons = [Neuron(nin) for _ in range(nout)]

	def __call__(self, x):
		if isinstance(x, Tensor):
			return self.forward_tensor(x)
		elif isinstance(x, list) and all(isinstance(neuron, Neuron) for neuron in x):
			return self.forward_neurons(x)
		else:
			raise ValueError(f"Unsupported input type: {type(x)}")

	def forward_tensor(self, x):
		outs = [n(x) for n in self.neurons]
		return outs[0] if len(outs) == 1 else outs

	def forward_neurons(self, x):
		outs = [n(neuron(x)) for n, neuron in zip(self.neurons, x)]
		return outs[0] if len(outs) == 1 else outs

	def parameters(self):
		return [p for neuron in self.neurons for p in neuron.parameters()]

	@staticmethod
	def actf(x, actf=Tensor.relu):
		return [actf(i) for i in x]
