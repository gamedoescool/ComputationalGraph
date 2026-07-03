import numpy as np

class TensorNode:
    def __init__(self,data: np.ndarray, is_param = True, dependents = None, forward_update = None, update_policy = None):
        self.data = data
        self.update_policy = update_policy
        self.forward_update = forward_update
        self.dependents = dependents

        if(self.update_policy == None):
            def dummy_update(lr,gradient):
                return
            self.update_policy = dummy_update
        if(forward_update == None):
            def dummy_update2():
                return
            self.forward_update = dummy_update2
        if(self.dependents == None):
            self.dependents = []
        self.gradient = np.zeros_like(data)
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
            self.gradient += gradient
            other.gradient += gradient
        def forward_update():
            return self.data + other.data
        return TensorNode(forward_update(), False, forward_update, update_policy)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rmul__(self,other):
        if(isinstance(other,float) == False):
            NotImplementedError("cannot multiply with " + other)
        def update_policy(gradient: np.ndarray):
            self.gradient += other*gradient
        def forward_update():
            return other*self.data
        return TensorNode(forward_update(), False, forward_update, update_policy)
    
    def __matmul__(self, other):
        if(isinstance(other,TensorNode) == False):
            NotImplementedError("cannot multiply with nontensor")
        def update_policy(lr:float, gradient: np.ndarray):
            #gradient * self @ other = gradient @ other.T * self = self.T @ gradient * other
            self.gradient += gradient @ other.data.T
            other.gradient += self.data.T @ gradient
        def forward_update():
            return self.data @ other.data
        return TensorNode(forward_update(), False, forward_update, update_policy)
    


    #TODO:
    #add activation function
    #add compile(), compute(np.ndarray input), train(), update() methods.
    
    
#interesting engineering thought: the way ive implemented it here makes it basically do depth first topologically backwards. this allows for the same node to perform backprop multiple times. 
#I think this allows for better model stability as the model takes more microsteps. I'll keep this and see how it performs.

#i mean i could implement the regular backprop technique and allow batching but Im deciding not to because i want to try out this novel approach.