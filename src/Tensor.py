from __future__ import annotations
from collections.abc import Callable
import numpy as np
from collections import deque
import utility as util
class TensorNode:
    def __init__(self,data: np.ndarray, dependents: set[TensorNode] | None = None, forward_update: Callable[[],np.ndarray] | None = None, update_policy: Callable[[np.ndarray],None] | None = None):
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
        self.temp_grad: np.ndarray = np.zeros_like(data)
        self.out_degree: int = 0 
    
    def update_data(self, new: np.ndarray):
        """
        Updates the data variable of the tensorNode. 
        Args:
            new (np.ndarray): the new ndarray value
        """
        self.data = new
        self.temp_grad = np.zeros_like(self.data)
        
    def refresh(self):
        """
        Refreshes the tensorNode, updates it.
        """
        self.data = self.forward_update()
        self.temp_grad = np.zeros_like(self.data)

    def backprop(self) -> None:
        """
        Performs Backprop one time, meaning it passes on the gradients to the creator tensor(s).
        """
        self.update_policy(self.temp_grad)
    
    def append_gradient(self, value: np.ndarray) -> None:
        """
        Appends the given gradient to the total gradient storage.
        Args:
            value (np.ndarray): 
                The gradient to append.
        """
        self.temp_grad += value
    
    def __add__(self, other: TensorNode) -> TensorNode:
        """
        Overlays the Numpy operation of adding two ndarrays.
        Args:
            other (TensorNode): the other tensornode
        Returns:
            a TensorNode representing self + other
        """
        access = util.find_out_dependents(self,other)
        def update_policy(gradient: np.ndarray):
            self.append_gradient(util.condense(gradient,self.temp_grad.shape))
            if(isinstance(other,TensorNode)):
                other.append_gradient(util.condense(gradient,other.temp_grad.shape))
        def forward_update():
            return access[1] + access[2]
        return TensorNode(forward_update(), access[0], forward_update, update_policy)

    def __radd__(self, other) -> TensorNode:
        """
        Overlays the Numpy operation of adding two ndarrays.
        Args:
            other (TensorNode): the other tensornode
        Returns:
            a TensorNode representing other + self
        """
        return self + other
    def __sub__(self, other: TensorNode) -> TensorNode:
        """
        Overlays the Numpy operation of subtracting two ndarrays.
        Args:
            other (TensorNode): the other tensornode
        Returns:
            a TensorNode representing self - other
        """
        access = util.find_out_dependents(self,other)
        def update_policy(gradient: np.ndarray):
            self.append_gradient(util.condense(gradient,self.temp_grad.shape))
            if(isinstance(other,TensorNode)):
                other.append_gradient(util.condense(-gradient,other.temp_grad.shape))
        def forward_update():
            return access[1] - access[2]
        return TensorNode(forward_update(), access[0], forward_update, update_policy)
    
    def __mul__(self,other: TensorNode) -> TensorNode:

        """
        Overlays the Numpy operation of "multiplying" two ndarrays.
        Args:
            other (TensorNode): the other tensornode
        Returns:
            a TensorNode representing self * other
        """
        access = util.find_out_dependents(self,other)
        def forward_update():
            return access[1] * access[2]
        def update_policy(gradient: np.ndarray):
            self.append_gradient(util.condense(access[2]*gradient,self.temp_grad.shape))
            if(isinstance(other,TensorNode)):
                other.append_gradient(util.condense(access[1] * gradient, other.temp_grad.shape))
        return TensorNode(forward_update(),access[0],forward_update,update_policy)
            
    def __rmul__(self,other) -> TensorNode:
        """
        Overlays the Numpy operation of "multiplying" two ndarrays.
        Args:
            other (TensorNode): the other tensornode
        Returns:
            a TensorNode representing other * self
        """
        return self * other
    
    def matmul(thing, other) -> TensorNode:
        access = util.find_out_dependents(thing,other)
        def update_policy(gradient: np.ndarray):
            #gradient * self @ other = gradient @ other.T * self = self.T @ gradient * other
            if(isinstance(thing,TensorNode)):
                thing.append_gradient(util.condense(gradient @ access[2].swapaxes(-2,-1), thing.temp_grad.shape))
            if(isinstance(other,TensorNode)):
                other.append_gradient(util.condense(access[1].swapaxes(-2,-1) @ gradient, other.temp_grad.shape))
        def forward_update():
            return access[1] @ access[2]
        return TensorNode(forward_update(), access[0], forward_update, update_policy)
    def __matmul__(self, other: TensorNode) -> TensorNode:
        """
        Overlays the Numpy operation of matrix multiplying two ndarrays. Only works if both tensors have the same dimension: make sure to pad ur tensors!
        Args:
            other (TensorNode): the other tensornode
        Returns:
            a TensorNode representing self @ other
        """
        return TensorNode.matmul(self,other)


    def __rmatmul__(self, other) -> TensorNode:
        """
        Overlays the Numpy operation of matrix multiplying two ndarrays. Only works if both tensors have the same dimension: make sure to pad ur tensors!
        Args:
            other (TensorNode): the other tensornode
        Returns:
            a TensorNode representing other @ self
        """
        return TensorNode.matmul(other,self)

    def div(thing, other) -> TensorNode:
        access = util.find_out_dependents(thing,other)
        def forward_update() -> np.ndarray:
            return access[1]/access[2]
        def update_policy(gradient: np.ndarray) -> None:
            if(isinstance(thing, TensorNode)):
                thing.append_gradient(util.condense(gradient/access[2],thing.data.shape))
            if(isinstance(other, TensorNode)):
                other.append_gradient(util.condense(-gradient * access[1]/(access[2]**2),other.data.shape))
        return TensorNode(forward_update(),access[0],forward_update,update_policy)
    
    def __truediv__(self, other: TensorNode) -> TensorNode:
        """
        Performs numpy division on the two tensors. PLEASE make sure that other dosent have any zero elements (or close to 0)
        Args:
            other (TensorNode): The other tensorNode
        Returns:
            out: A tensornode representing the numpy division of self/other
        """
        return TensorNode.div(self,other)
    def __rtruediv__(self, other):
        """
        Performs numpy division on the two tensors. PLEASE make sure that other dosent have any zero elements (or close to 0)
        Args:
            other (TensorNode): The other tensorNode
        Returns:
            out: A tensornode representing the numpy division of self/other
        """
        return TensorNode.div(other,self)
    
    #burner method that will go once i get something more abstract working
    def get_index(self, j: int):
        self.out_degree += 1
        def forward_update():
            return self.data[j]
        def update_policy(gradient):
            self.temp_grad[j] += gradient #i am the alpha sigma omega and tanny at the same time :moai:
        return TensorNode(forward_update(),set([self]),forward_update,update_policy)

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

        
    

    

