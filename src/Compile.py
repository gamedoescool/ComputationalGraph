import Tensor as t
import numpy as np
from collections import deque
from operator import methodcaller

class Pipeline:
    def __init__(self, rev_topo_sort: list[list[t.TensorNode]]):
        self.rev_topo_sort: list[t.TensorNode]= [] 
        #flatten the whole structure like a sigma
        for level in rev_topo_sort:
            for node in level:
                self.rev_topo_sort.append(node)
    
    def backprop(self) -> None:
        """
        Accumulates the gradients in the pipeline to according to the scalar loss. Assumes the final layer is the loss function
        """
        if(self.rev_topo_sort[0].data.size != 1):
            raise RuntimeError("Final layer is not a scalar loss function. Please add the loss function")
        self.rev_topo_sort[0].temp_grad = np.ones_like(self.rev_topo_sort[0].temp_grad)

        #update paramaters
        for node in self.rev_topo_sort:
            node.backprop()

    def forward(self) -> None:
        """
        Updates the entire model, ensures all values are accurate. Also sets all gradients to 0 the hero. 
        """
        for node in self.rev_topo_sort:
            node.refresh()


