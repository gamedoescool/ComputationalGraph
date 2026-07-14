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
X = np.random.uniform(0,1,size=(1,10))
Y = np.random.uniform(0,1,size=(1,15))

x = t.TensorNode(X,is_param=False) #2
x_new = op.copy(x)
W = t.TensorNode(w) 
B = t.TensorNode(b) 
Prod = x_new @ W
L = Prod + B 
Y_out = (op.sAct(L))
summyPack = op.sum(Y_out,axis=(1),keepDim=True)
Y_curr = t.TensorNode(Y,is_param=False)
Y_new = Y_out / (summyPack + 1e-7)
diff = Y_curr-Y_new
loss = op.norm_squared(diff)

#2 + 1 + 1 + 2 + 1 + 3 = 10



compiler = (loss.compile())

list = compiler
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
    summyPack:"sum(Y)",
    Y_new:"Y_reg",
    diff: "f(L)-Y_reg",
    loss: "||diff||"
   
}
print(Y_new.data[0])
print(f"loss {loss.data}")
compiler[0][0].temp_grad = np.ones_like(compiler[0][0].temp_grad)

for level in compiler:
        for node in level:
            node.backprop()
    
for level in reversed(compiler):
    for node in level:
        node.update_params(3)
print(f"loss {loss.data}")

print(Y_new.data[0])



