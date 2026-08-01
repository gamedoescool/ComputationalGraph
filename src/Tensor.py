from __future__ import annotations
from collections.abc import Callable
import numpy as np
from collections import deque
import utility as util
class TensorNode:
    def __init__(self,data: np.ndarray, is_param: bool = True, dependents: set[TensorNode] | None = None, forward_update: Callable[[],np.ndarray] | None = None, update_policy: Callable[[np.ndarray],None] | None = None):
        #actual tensor data
        self.data: np.ndarray = data
        #linking tools
        self.update_policy: Callable[[np.ndarray],None] | None = update_policy
        self.forward_update: Callable[[],np.ndarray] | None = forward_update
        self.dependents: set[TensorNode] | None = dependents

        #default cases
        if(self.update_policy is None):
            def dummy_update(gradient):
                return
            self.update_policy = dummy_update
        if(forward_update is None):
            def dummy_update2():
                return self.data
            self.forward_update = dummy_update2
        
        if(self.dependents is None):
            self.dependents = set([])

        #technical gradient components
        if(is_param):
            self.gradient: np.ndarray = np.zeros_like(data)
        self.temp_grad: np.ndarray = np.zeros_like(data)
        self.is_param: bool = is_param
        self.out_degree: int = 0 
    
    def update_data(self, new: np.ndarray):
        """
        Updates the data variable of the tensorNode. Only works if the tensorNode is not a paramater.
        Args:
            new (np.ndarray): the new ndarray value
        """
        if(self.is_param):
            raise Exception("Cannot manually update the data of a paramater.")
        self.data = new
        self.update_gradient_init()
    def update_gradient_init(self):
        """
        Updated the gradients to be a 0 like data
        """
        self.temp_grad = np.zeros_like(self.data)
        if self.is_param:
            self.gradient = np.zeros_like(self.data)
        
    def refresh(self):
        """
        Refreshes the tensorNode, updates it.
        """
        self.data = self.forward_update()
        self.update_gradient_init()
    def update_params(self, lr: float) -> None:
        """
        Updates the data variable by either performing gradient descent or making it match dependents.
        Args:
            expression (np.ndarray): the expression at hand
        """
        if(self.is_param):
            self.data -= lr*self.gradient
            self.gradient = np.zeros_like(self.data)
        else:
            self.data = self.forward_update()

    def backprop(self) -> None:
        """
        Performs Backprop one time, meaning it passes on the gradients to the creator tensor(s). It also accumulates total gradients

        """
        if(self.is_param):
            self.gradient += self.temp_grad
        self.update_policy(self.temp_grad)
        self.temp_grad = np.zeros_like(self.data)
    
    def append_gradient(self, value: np.ndarray) -> None:
        """
        Appends the given gradient to the total gradient storage.
        Args:
            value (np.ndarray): 
                The gradient to append.
        """
        self.temp_grad += value
    
    #TODO: once toposort works make temp_grad += to allow for duplicates and all that jazz
    def __add__(self, other: TensorNode | float) -> TensorNode:
        """
        Overlays the Numpy operation of adding two ndarrays.
        Args:
            other (TensorNode): the other tensornode
        Returns:
            a TensorNode representing self + other
        """
        if(isinstance(other,TensorNode) == False and isinstance(other,float) == False):
            raise NotImplementedError("cannot add with type " + str(type(other)))
        self.out_degree += 1
        other.out_degree += 1
        def update_policy(gradient: np.ndarray):
            self.append_gradient(util.condense(gradient,self.temp_grad.shape))
            other.append_gradient(util.condense(gradient,other.temp_grad.shape))
        def forward_update():
            return self.data + other.data
        return TensorNode(forward_update(), False, set([self,other]), forward_update, update_policy)

    def __sub__(self, other: TensorNode) -> TensorNode:
        """
        Overlays the Numpy operation of subtracting two ndarrays.
        Args:
            other (TensorNode): the other tensornode
        Returns:
            a TensorNode representing self - other
        """
        if(isinstance(other,TensorNode) == False and isinstance(other,float) == False):
            raise NotImplementedError("cannot subtract with type " + str(type(other)))
        if(isinstance(other,float)):
            other = TensorNode(np.zeros(shape=(1))+other,False)
        self.out_degree += 1
        other.out_degree += 1
        def update_policy(gradient: np.ndarray):
            self.append_gradient(util.condense(gradient,self.temp_grad.shape))
            other.append_gradient(util.condense(-gradient,other.temp_grad.shape))
        def forward_update():
            return self.data - other.data
        return TensorNode(forward_update(), False, set([self,other]), forward_update, update_policy)
    
    def __mul__(self,other: TensorNode) -> TensorNode:

        """
        Overlays the Numpy operation of "multiplying" two ndarrays.
        Args:
            other (TensorNode): the other tensornode
        Returns:
            a TensorNode representing self * other
        """
        self.out_degree += 1
        other.out_degree += 1
        if(isinstance(other,float) == False):
            NotImplementedError("cant do mult with " + str(type(other)))
        def forward_update():
            return self.data * other.data
        def update_policy(gradient: np.ndarray):

            self.append_gradient(util.condense(other.data*gradient,self.temp_grad.shape))
            other.append_gradient(util.condense(self.data * gradient, other.temp_grad.shape))
        return TensorNode(forward_update(),False,set([self,other]),forward_update,update_policy)
            
    def __rmul__(self,other: TensorNode) -> TensorNode:
        """
        Multiples this tensor by scalar (kinda useless might depreciate)
        Args:
            other (TensorNode): the other tensornode
        Returns:
            a TensorNode representing other * self
        """
        self.out_degree += 1
        other.out_degree += 1
        if(isinstance(other,float) == False):
            NotImplementedError("cannot multiply with " +  str(other))
        def update_policy(gradient: np.ndarray):
            self.append_gradient(other * gradient)
        def forward_update():
            return other*self.data
        return TensorNode(forward_update(), False, set([self]), forward_update, update_policy)
    
    def __matmul__(self, other: TensorNode) -> TensorNode:
        """
        Overlays the Numpy operation of matrix multiplying two ndarrays.
        Args:
            other (TensorNode): the other tensornode
        Returns:
            a TensorNode representing self @ other
        """
        self.out_degree += 1
        other.out_degree += 1
        if(isinstance(other,TensorNode) == False):
            NotImplementedError("cannot multiply with nontensor")
        def update_policy(gradient: np.ndarray):
            #gradient * self @ other = gradient @ other.T * self = self.T @ gradient * other
            self.append_gradient(util.condense(gradient @ other.data.swapaxes(-2,-1), self.temp_grad.shape))
            other.append_gradient(util.condense(self.data.swapaxes(-2,-1) @ gradient, other.temp_grad.shape))
        def forward_update():
            return self.data @ other.data
        return TensorNode(forward_update(), False, set([self,other]), forward_update, update_policy)
    
    def __truediv__(self, other: TensorNode) -> TensorNode:
        """
        Performs numpy division on the two tensors. PLEASE make sure that other dosent have any zero elements (or close to 0)
        Args:
            other (TensorNode): The other tensorNode
        Returns:
            out: A tensornode representing the numpy division of self/other
        """
        self.out_degree += 1
        other.out_degree += 1
        def forward_update() -> np.ndarray:
            return self.data/other.data
        def update_policy(gradient: np.ndarray) -> None:
            self.append_gradient(util.condense(gradient/other.data,self.data.shape))
            other.append_gradient(util.condense(-gradient * self.data/(other.data * other.data),other.data.shape))
        return TensorNode(forward_update(),False,set([self,other]),forward_update,update_policy)

    def compile(self) -> list[list[TensorNode]]:
        """
        Compiles the entire pipeline, allows for traning.
        Returns:
            a reverse topological ordering of the entire deep learning pipeline"""
        #okay, this should work....

        rev_topo_sort: list[list[TensorNode]] = []
        iterator: deque[TensorNode] = deque()
        iterator.append(self)
        curr: int = 0
        total: int = 1
        newTotal: int = 0
        level: list[TensorNode] = []
        #modified toposort logic to allow layering
        while(len(iterator) != 0):
            current = iterator.popleft()
            if(curr == total):
                rev_topo_sort.append(level)
                total = newTotal
                newTotal = 0
                level = []
                curr = 0
            level.append(current)
            for depend in current.dependents:
                depend.out_degree -= 1
                if(depend.out_degree == 0):
                    iterator.append(depend)
                    newTotal+=1
            curr+=1
        if(len(level) != 0):
            rev_topo_sort.append(level)
        #now re add indegrees in case we need them
        for layer in rev_topo_sort:
            for node in layer:
                for depend in node.dependents:
                    depend.out_degree += 1
        return (rev_topo_sort)

        
    

    

