import numpy as np
from tqdm import trange

def accuracy(pred, true):
    return np.sum(pred == true) / len(true)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, X):
        self.output = np.dot(X, self.weights) * self.biases

class Activation_Relu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probs

# stochastic gradient descent
'''
class Adam:
    def __init__(self, bt1=0.9, bt2=0.999, eps=10e-8, lr=1e-3):
        self.lr = lr
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0
        self.bt1 = bt1
        self.bt2 = bt2
        self.eps = eps

    def zero_grad(self):
        # this is also wrong! need to zero they gradients on all the neurons!
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0

    def step(self, w, b, dw, db):
        self.m_dw = self.b1*self.m_dw + (1 - self.bt1)*dw
        self.m_db = self.bt1*self.m_db + (1 - self.bt1)*db

        self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*(dw**2)
        self.v_db = self.beta2*self.v_db + (1-self.beta2)*(db)

        m_dw_corr = self.m_dw/(1-self.beta1**t)
        m_db_corr = self.m_db/(1-self.beta1**t)
        v_dw_corr = self.v_dw/(1-self.beta2**t)
        v_db_corr = self.v_db/(1-self.beta2**t)

        w = w - self.eta*(m_dw_corr/(np.sqrt(v_dw_corr)+self.epsilon))
        b = b - self.eta*(m_db_corr/(np.sqrt(v_db_corr)+self.epsilon))

        return w, b
'''

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        return data_loss

# generally calculates the difference from actual and predicted value
class Loss_CatergoricalCrossentropy(Loss):
    # forward pass simply gets the loss of the entire model
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1: # mean scalar value were passed
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)

        return negative_log_likelihoods

    # the backward pass of the loss function gives each neuron in the model a gradient
    #   that the optimzer can then use to tune each neuron in the right direction
    #def backward(self):

class NN:
    def __init__(self):
        # 1 hidden layer
        self.h1 = Layer_Dense(784, 128)
        self.a1 = Activation_Relu()
        self.h2 = Layer_Dense(128, 128)
        self.a2 = Activation_Softmax()

    def forward(self, X):
        self.h1.forward(X)
        self.a1.forward(self.h1.output)
        self.h2.forward(self.a1.output)
        self.a2.forward(self.h2.output)

        return self.a2.output

if __name__ == "__main__":
    from keras.datasets import mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    model = NN()
    loss_fn = Loss_CatergoricalCrossentropy()
    #optimizer = Adam(lr=1e-4)

    epochs = 2
    batch_size = 5

    for epoch in (t := trange(epochs)):
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            Y_batch = Y_train[i:i+batch_size]

            out = model.forward(X_batch)
            #optimizer.zero_grad()
            loss = loss_fn.calculate(out, Y_batch)
            #loss.backward()
            #optimizer.step()

        t.set_description("loss %.2f" % (loss))
