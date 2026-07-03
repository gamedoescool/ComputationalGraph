import numpy as np
from collections import deque

class TensorNode:
    def __init__(self,data: np.ndarray, is_param = True, dependents = None, forward_update = None, update_policy = None):
        self.data = data
        self.update_policy = update_policy
        self.forward_update = forward_update
        self.dependents = dependents

        if(self.update_policy == None):
            def dummy_update(gradient):
                return
            self.update_policy = dummy_update
        if(forward_update == None):
            def dummy_update2():
                return
            self.forward_update = dummy_update2
        if(self.dependents == None):
            self.dependents = []
        self.gradient = np.zeros_like(data)
        self.temp_grad = np.zeros_like(data)
        self.is_param = is_param
        self.compiled = False
        self.topo_sort = []
    
    def backprop(self, outerGrad: np.ndarray):
        if(outerGrad.shape != self.data.shape):
            SyntaxError("Gradient must be same shape as the data. This should not happen unless something went very wrong on my side.")
        self.update_policy(outerGrad)
    
    def update_params(self, lr):
        if(self.is_param):
            self.data -= lr*self.gradient
        self.gradient -= self.gradient

    
    def __add__(self, other):
        if(isinstance(other,TensorNode) == False):
            NotImplementedError("cannot add with nontensor")
        def update_policy(gradient: np.ndarray):
            self.temp_grad = gradient
            self.gradient += self.temp_grad

            other.temp_grad = gradient
            other.gradient += other.temp_grad
        def forward_update():
            return self.data + other.data
        return TensorNode(forward_update(), False, [self,other], forward_update, update_policy)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rmul__(self,other):
        if(isinstance(other,float) == False):
            NotImplementedError("cannot multiply with " + other)
        def update_policy(gradient: np.ndarray):
            self.temp_grad = other * gradient
            self.gradient += other*self.temp_grad
        def forward_update():
            return other*self.data
        return TensorNode(forward_update(), False, [self], forward_update, update_policy)
    
    def __matmul__(self, other):
        if(isinstance(other,TensorNode) == False):
            NotImplementedError("cannot multiply with nontensor")
        def update_policy(gradient: np.ndarray):
            #gradient * self @ other = gradient @ other.T * self = self.T @ gradient * other
            self.temp_grad = gradient @ other.data.T
            self.gradient += self.temp_grad

            other.temp_grad = self.data.T @ gradient
            other.gradient += other.temp_grad
        def forward_update():
            return self.data @ other.data
        return TensorNode(forward_update(), False, [self,other], forward_update, update_policy)
    
    def dot(self,other):
        def update_policy(gradient: np.ndarray):
            self.temp_grad = gradient*other.data
            self.gradient += self.temp_grad
            other.temp_grad = gradient*self.data
            other.gradient += other.temp_grad
        def forward_update():
            return np.sum(self.data * other.data)
        return TensorNode(forward_update(), False, [self, other], forward_update, update_policy)
    def compile(self):
        self.iterator = deque()
        if(self.data.size != 1):
            TypeError("Tensor MUST be a scalar in order to compile the pipeline")
        self.iterator.append([self])

        while(len(self.iterator) != 0):
            current = self.iterator.popleft()
            if(len(current) != 0):
                self.topo_sort.append(current)
                new_depend = []
                for node in current:
                    new_depend += (node.dependents)
                self.iterator.append(new_depend)

        #gradient prepping
        self.temp_grad = np.ones_like(self.data)

        #debug return
        self.compiled = True
        return self.topo_sort
    
    def train(self):
        if(self.data.size != 1):
            TypeError("Tensor MUST be a scalar in order to train the pipeline")
        if (self.compiled == False):
            RuntimeError("Pipeline must be compiled in order to train")
        for level in self.topo_sort:
            #parallel magic maybe?
            for node in level:
                node.update_policy(node.temp_grad)

    def update(self, lr: float):
        if(self.data.size != 1):
            TypeError("Tensor MUST be a scalar in order to update the pipeline")
        if (self.compiled == False):
            RuntimeError("Pipeline must be compiled in order to update")
        for level in self.topo_sort:
            #parallel magic maybe?
            for node in level:
                node.data -= lr*node.gradient
                node.gradient -= node.gradient

    #TODO:
    #add activation function
    #add compile(), compute(np.ndarray input), train(), update() methods.

    
#interesting engineering thought: the way ive implemented it here makes it basically do depth first topologically backwards. this allows for the same node to perform backprop multiple times. 
#I think this allows for better model stability as the model takes more microsteps. I'll keep this and see how it performs.

#i mean i could implement the regular backprop technique and allow batching but Im deciding not to because i want to try out this novel approach.