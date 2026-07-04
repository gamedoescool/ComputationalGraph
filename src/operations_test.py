import Tensor as t
import numpy as np


#NOTE: if u wanna do a operator twice (like z dot z z @ z z + z) u should use .copy() for one of the args


#if u dont, gradients could incorrectly update. 
x_1 = np.random.uniform(0,1,size=(10,1))
y_1 = np.random.uniform(0,1,size=(3,10))

x = t.TensorNode(x_1)
y = t.TensorNode(y_1)


z = y @ x
print(z.data)
print((y_1 @ x_1))


def forward_update(input: np.ndarray):
            return 1/(1+np.exp(-input))

print(forward_update(x_1))