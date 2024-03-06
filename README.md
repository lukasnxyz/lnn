## lnn
A minimal deep learning library inspired by tinygrad or micrograd. The goal of this library is to
make the building and modeling of neural networks extremely easy and minimal. (After getting some
base functionalities, I do plan on trying to make this an extremely efficient library as well with
accelerators and so forth.) For now, this is simply just a toy project for me to learn with.

### Quick start
Setup virtual enviornment
```bash
cd lnn/
python3 -m venv tf-virtual-env
pip3 install keras tensorflow
source nn-virtual-env/bin/activate
```

```python
# None yet
```

### Components
1. Very simple Tensor class (Tensor, operations, autograd)
2. Basic layer functionalities (Neuron, layer, MLP)
3. Minimal functions (Cross-entropy loss, back propagation)
