import numpy as np
from collections import deque
import utility as util
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
                return self.data
            self.forward_update = dummy_update2
        if(self.dependents == None):
            self.dependents = []
        
        self.gradient = np.zeros_like(data)
        self.temp_grad = np.zeros_like(data)
        self.is_param = is_param
    
    def update_params(self, lr):
        if(self.is_param):
            self.data -= lr*self.gradient
        self.gradient -= self.gradient

    #TODO: once toposort works make temp_grad += to allow for duplicates and all that jazz
    def __add__(self, other):
        if(isinstance(other,TensorNode) == False):
            NotImplementedError("cannot add with nontensor")
        def update_policy(gradient: np.ndarray):
            k = np.zeros_like(self.gradient)
            util.condense(gradient,k.shape,k)
            self.temp_grad = k
            self.gradient += self.temp_grad

            other.temp_grad = k
            other.gradient += other.temp_grad
        def forward_update():
            return self.data + other.data
        return TensorNode(forward_update(), False, [self,other], forward_update, update_policy)
    
    def __sub__(self, other):
        return self + (-1 * other)
    
    def __mul__(self,other):
        def forward_update():
            return self.data * other.data

        def update_policy(gradient: np.ndarray):
            k = np.zeros_like(self.data)
            util.condense(other.data*gradient,k,out=k)
            self.temp_grad = k
            self.gradient += self.temp_grad

            k = np.zeros_like(other.data)
            util.condense(self.data * gradient, k, out=k)
            other.temp_grad = k
            other.gradient += other.temp_grad
        return TensorNode(forward_update(),False,[self,other],forward_update,update_policy)
            
    def __rmul__(self,other):
        if(isinstance(other,float) == False):
            NotImplementedError("cannot multiply with " +  str(other))
        def update_policy(gradient: np.ndarray):
            np.multiply(other,gradient,out=self.temp_grad)
            self.gradient += other*self.temp_grad
        def forward_update():
            return other*self.data
        return TensorNode(forward_update(), False, [self], forward_update, update_policy)
    
    def __matmul__(self, other):
        if(isinstance(other,TensorNode) == False):
            NotImplementedError("cannot multiply with nontensor")
        def update_policy(gradient: np.ndarray):
            #gradient * self @ other = gradient @ other.T * self = self.T @ gradient * other
            np.matmul(gradient, other.data.T, out=self.temp_grad)
            self.gradient += self.temp_grad
            
            np.matmul(self.data.T,gradient,out=other.temp_grad)
            other.gradient += other.temp_grad
        def forward_update():
            return self.data @ other.data
        return TensorNode(forward_update(), False, [self,other], forward_update, update_policy)
    
    def dot(self,other):
        def update_policy(gradient: np.ndarray):
            np.multiply(gradient,other.data,out=self.temp_grad)
            self.gradient += self.temp_grad
            np.multiply(gradient,self.data,out=other.temp_grad)
            other.gradient += other.temp_grad
        def forward_update():
            return np.sum(self.data * other.data)
        return TensorNode(forward_update(), False, [self, other], forward_update, update_policy)
    
    def norm_squared(self):
        def update_policy(gradient: np.ndarray):
            self.temp_grad = 2*(gradient*self.data)
            self.gradient += self.temp_grad
        def forward_update():
            return np.sum(self.data * self.data)
        return TensorNode(forward_update(), False, [self], forward_update, update_policy)


    def copy(self):
        def update_policy(gradient: np.ndarray):
            self.temp_grad = gradient
            self.gradient += self.temp_grad
        def forward_update():
            return self.data
        return TensorNode(forward_update(), False, [self], forward_update, update_policy)
    
    
    def sAct(self):
        def forward_update():
            return util.sAct(self.data)
        def update_policy(gradient: np.ndarray):
            a = util.sActPrime(self.data)
            self.temp_grad = gradient * a
            self.gradient += self.temp_grad
        return TensorNode(forward_update(), False, [self], forward_update, update_policy)
    
    #we NEED to make this work on a PER ROW basis...
    def regularization(self, epsilon):
        def forward_update():
            return self.data/(np.sum(self.data)+epsilon)
        def update_policy(gradient: np.ndarray):
            denom = np.sum(self.data)+epsilon
            val1 = denom * gradient
            val2 = np.ones_like(self.data) * (np.sum(gradient * self.data))
            self.temp_grad = (val1 - val2)/(denom*denom)
            self.gradient += self.temp_grad
        return TensorNode(forward_update(), False, [self], forward_update, update_policy) 

    #dp/dt = kp(1-p/L) = p(1-p) as k = 1 and L = 1
    def sigmoid(self):
        def forward_update():
            return 1/(1+np.exp(-self.data)) # i love numpy broadcasting
        def update_policy(gradient: np.ndarray):
            dx = forward_update()*(1-forward_update)
            self.temp_grad = gradient * dx
            self.gradient += self.temp_grad
        return TensorNode(forward_update(),False,[self],forward_update,update_policy)

    def compile(self):
        topo_sort = []
        iterator = deque()
        iterator.append([self])

        while(len(iterator) != 0):
            current = iterator.popleft()
            if(len(current) != 0):
                topo_sort.append(current)
                new_depend = []
                for node in current:
                    new_depend += (node.dependents)
                iterator.append(new_depend)

        #gradient prepping
        self.temp_grad = np.ones_like(self.data)
        return Pipeline(topo_sort)

        
    #TODO:
    #fix toposort to actually toposort
    
class Pipeline:
    def __init__(self, topo_sort):
        self.topo_sort = topo_sort   
    
    def train(self):
        #update paramaters
        self.recompute()
        for level in self.topo_sort:
            #parallel magic maybe?
            for node in level:
                node.update_policy(node.temp_grad)
                #TODO: once toposort works set node.temp_grad to 0

    def update(self, lr: float):
        for level in self.topo_sort:
            for node in level:
                node.update_params(lr)
        
    def recompute(self):
        for layer in reversed(self.topo_sort):
            for node in layer:
                node.data = node.forward_update()
    def update_input(self, input):
        if(len(self.topo_sort[-1]) != 1):
            raise RuntimeError("Initial layer is ambiguous, please use .initalize()")
        self.topo_sort[-1][0] = input


