import numpy as np

class TensorNode:
    def __init__(self,data: np.ndarray, is_param = True, update_policy = None):
        self.data = data
        self.update_policy = update_policy
        if(self.update_policy == None):
            def dummy_update(lr,gradient):
                return
            self.update_policy = dummy_update
        
        self.gradient = np.zeros_like(data)
        self.is_param = is_param
    
    def backprop(self,lr:float, outerGrad: np.ndarray):
        if(outerGrad.shape != self.data.shape):
            SyntaxError("Gradient must be same shape as the data. This should not happen unless something went very wrong on my side.")
        if(self.is_param):
            self.data -= lr*outerGrad
        self.update_policy(lr, outerGrad)
    
    def __add__(self, other):
        if(isinstance(other,TensorNode) == False):
            NotImplementedError("cannot add with nontensor")
        def update_policy(lr:float, gradient: np.ndarray):
            self.backprop(lr,gradient)
            other.backprop(lr,gradient)
        return TensorNode(self.data + other.data, False, update_policy)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rmul__(self,other):
        if(isinstance(other,float) == False):
            NotImplementedError("cannot multiply with " + other)
        def update_policy(lr:float, gradient: np.ndarray):
            self.backprop(lr,other*gradient)
        return TensorNode(self.data, False, update_policy)
    
    def __matmul__(self, other):
        if(isinstance(other,TensorNode) == False):
            NotImplementedError("cannot multiply with nontensor")
        def update_policy(lr:float, gradient: np.ndarray):
            #gradient * self @ other = gradient @ other.T * self = self.T @ gradient * other
            self.backprop(lr, gradient @ other.data.T)
            other.backprop(lr, self.data.T @ gradient)
        return TensorNode(self.data @ other.data, False, update_policy)

    #TODO:
    #add activation function
    #i guess go with batch based method because this is a uphill battle
    
#interesting engineering thought: the way ive implemented it here makes it basically do depth first topologically backwards. this allows for the same node to perform backprop multiple times. 
#I think this allows for better model stability as the model takes more microsteps. I'll keep this and see how it performs.

#i mean i could implement the regular backprop technique and allow batching but Im deciding not to because i want to try out this novel approach.