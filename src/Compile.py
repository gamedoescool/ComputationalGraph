import Tensor as t
import numpy as np
class Pipeline:
    def __init__(self, rev_topo_sort: list[list[t.TensorNode]]):
        self.rev_topo_sort = rev_topo_sort   
    
    def train(self) -> None:
        """
        Accumulates the gradients in the pipeline to according to the scalar loss. Assumes the final layer is the loss function
        """
        if(self.rev_topo_sort[0][0].data.size != 1):
            raise RuntimeError("Final layer is not a scalar loss function. Please add the loss function")
        self.rev_topo_sort[0][0].temp_grad = np.ones_like(self.rev_topo_sort[0][0].temp_grad)

        #update paramaters
        for level in self.rev_topo_sort:
            #parallel magic maybe?
            for node in level:
                node.backprop()

    def update(self, lr: float) -> None:
        """
        Updates the parameters of the pipeline and recomputes all associated Tensors
        Args:
            lr (float): Learning Rate to be used for Gradient Descent
        """
        for layer in reversed(self.rev_topo_sort):
            for node in layer:
                node.update_params(lr)
    

    def update_input(self, input: np.ndarray) -> None:
        """
        Updates the entire model based on the changed input
        """
        if(len(self.rev_topo_sort[-1]) != 1):
            raise RuntimeError("Initial layer is ambiguous, please use .initalize() before compiling")
        self.rev_topo_sort[-1][0].data = input
        for layer in reversed(self.rev_topo_sort):
            for node in layer:
                node.data = node.forward_update()