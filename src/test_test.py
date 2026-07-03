import Tensor as t
import numpy as np


def pipeline(x_1,y_1,train):
    x = t.TensorNode(x_1,True)
    y = t.TensorNode(y_1,True)

    z = (y @ x)
    #lets try to minimize z^2 (z dot z) guys!
    loss = z.dot(z)


    print(loss.data)
    if(train):
        k = loss.compile()
        loss.train()
        loss.update(0.003)
    


x_1 = np.random.uniform(0,1,size=(10,1))
y_1 = np.random.uniform(0,1,size=(3,10))

print(x_1)
print(y_1)
pipeline(x_1,y_1,True)
print(x_1)
print(y_1)
pipeline(x_1,y_1,False)


# print(z.data.T @ z.data)

#it actually works... now what?

#we redo the mnist dataset but 900 times better 🗿