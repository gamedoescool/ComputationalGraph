import Tensor as t
import numpy as np

#simple perceptron example (works like a CHARM)
epsilon = 1e-7
def func(input: np.ndarray):
    mask = (input >= 0)
    mask1 = (input < 0)
    a = np.zeros_like(input)
    a[mask] = input[mask]+1
    a[mask1] = np.exp(input[mask1])
    return a

#weights
w = np.random.uniform(0,1,size=(10,15))
b = np.random.uniform(0,1,size=(1,15))

#demo input output
X = np.random.uniform(0,1,size=(60_000,10))
Y = np.random.uniform(0,1,size=(60_000,15))

x = t.TensorNode(X,is_param=False).copy()

W = t.TensorNode(w)
B = t.TensorNode(b)

L = x @ W + B
Y_out = L.sAct()

loss = (t.TensorNode(Y,is_param=False) - Y_out).norm_squared()

l = func(X @ w + b) - Y

print(np.sum(l*l))
print(loss.data)

compiler = loss.compile()
compiler.train()
compiler.update(0.003)
compiler.train()
compiler.update(0.003)
compiler.train()

print(loss.data)
l = func(X @ w + b) - Y

print(np.sum(l*l))