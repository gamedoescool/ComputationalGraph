import Tensor as t
import numpy as np
x = np.random.uniform(0,1,size=(67,1))
y = np.random.uniform(0,1,size=(3,67))

x = t.TensorNode(x,True)
y = t.TensorNode(y,True)

z = (y @ x).act()
#lets try to minimize z^2 (z dot z) guys!
print(z.data.T @ z.data)
for i in range(10):
    z.backprop(0.007/(i*i+2*i+1),2*z.data)
    z = (y @ x).act()
    print(z.data.T @ z.data)


# print(z.data.T @ z.data)

#it actually works... now what?

#we redo the mnist dataset but 900 times better 🗿