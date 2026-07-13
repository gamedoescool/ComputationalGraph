import Tensor as t
import numpy as np
import Operations as op
import Compile as c
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

x = t.TensorNode(X,is_param=False) #2
x_new = op.copy(x)
W = t.TensorNode(w) 
B = t.TensorNode(b) 
Prod = x_new @ W
L = Prod + B 
Y_out = (op.sAct(L))
Y_curr = t.TensorNode(Y,is_param=False)
diff = Y_out-Y_curr
loss = op.norm_squared(diff) 

#2 + 1 + 1 + 2 + 1 + 3 = 10



compiler = c.Pipeline(loss.compile())

list = compiler.rev_topo_sort
print(len(list))
ohiogyat = {
    x:"x",
    x_new:"x.copy()",
    W: "W",
    B: "B",
    Prod:"x.copy() @ W",
    L:"L",
    Y_out:"Y_out = f(L)",
    Y_curr:"Y",
    diff: "f(L)-Y",
    loss: "||diff||"
   
}
# print(Y_out.data[0])
# print(f"loss {loss.data}")

# compiler.train()
# compiler.update(0.25)
# print(f"loss {loss.data}")
# compiler.train()
# compiler.update(0.25)
# print(f"loss {loss.data}")

# print(Y_out.data[0])

z = Y/np.sum(Y,axis=(1),keepdims=True)

print(np.sum(z[32]))

