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

print('--------------------------------------')
random_grad_mat = np.random.uniform(0,1,size=(3,1))

z.update_policy(np.random.uniform(0,1,size=(3,1)))

print(x.gradient)

print(y.gradient)