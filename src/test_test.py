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

x = t.TensorNode(X,is_param=False) #2
x_new = x.copy()
W = t.TensorNode(w) 
B = t.TensorNode(b) 
Prod = x_new @ W
L = Prod + B 
Y_out = L.sAct() 
Y_curr = t.TensorNode(Y,is_param=False)
diff = Y_out-Y_curr
loss = diff.norm_squared() 

#2 + 1 + 1 + 2 + 1 + 3 = 10



compiler = loss.compile()

list = compiler.topo_sort

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

for level in list:
    ohio = str(len(level))
    for k in level:
        ohio += " (" + (ohiogyat[k]) + ")" #if this work i am the greatest
    print(ohio)

