import Tensor as t
import numpy as np
import utility as util
def dot(self: t.TensorNode,other: t.TensorNode) -> t.TensorNode:
    """
    Computes the generalized dot product between the two Tensors
    Args:
        self (TensorNode): first tensor
        other (TensorNode): second tensor
    Returns:
        The generalized dot product self dot other
    """
    self.out_degree += 1
    other.out_degree += 1
    def update_policy(gradient: np.ndarray):
        self.append_gradient(gradient * other.data)
        other.append_gradient(gradient * self.data)
    def forward_update():
        return np.sum(self.data * other.data)
    return t.TensorNode(forward_update(), False, set([self, other]), forward_update, update_policy)
    
def norm_squared(self: t.TensorNode) -> t.TensorNode:
    """
    Computes the generalized "magnitude" or norm squared of the tensor
    Args:
        self (TensorNode): the tensor
    Returns:
        The generalized norm squared sqrt(self dot self)
    """
    self.out_degree += 1
    def update_policy(gradient: np.ndarray):
        self.append_gradient(2*(gradient*self.data))
    def forward_update():
        return np.sum(self.data * self.data)
    return t.TensorNode(forward_update(), False, set([self]), forward_update, update_policy)


def copy(self: t.TensorNode) -> t.TensorNode:
    """
    Returns a identical copy of the tensorNode. Copies the data by reference as well.
    Args:
        self (TensorNode): the tensor
    Returns:
        A copy of the node.
    """
    self.out_degree += 1
    def update_policy(gradient: np.ndarray):
        self.append_gradient(gradient)
    def forward_update():
        return self.data
    return t.TensorNode(forward_update(), False, set([self]), forward_update, update_policy)


def sAct(self: t.TensorNode) -> t.TensorNode:
    """
    Performs the custom sAct activation function on each element. 
    Args:
        self (TensorNode): the tensor
    Returns:
        The sAct of the tensor sAct(self)
    """
    self.out_degree += 1
    def forward_update():
        return util.sAct(self.data)
    def update_policy(gradient: np.ndarray):
        a = util.sActPrime(self.data)
        self.append_gradient(gradient * a)
    return t.TensorNode(forward_update(), False, set([self]), forward_update, update_policy)

#we NEED to make this work on a PER ROW basis...
def regularization(self: t.TensorNode, epsilon: float = 1e-7) -> t.TensorNode:
    """
    Regulaizes the tensorNode by dividing by the total sum of all elements, plus a error term. 
    Args:
        self (TensorNode): the tensor. Please ensure the tensor is positive
        epsilon (float): the error term preventing division by 0.
    Returns:
         the tensor self/(sum(self) + epsilon)
    """
    self.out_degree += 1
    def forward_update():
        return self.data/(np.sum(self.data)+epsilon)
    def update_policy(gradient: np.ndarray):
        denom = np.sum(self.data)+epsilon
        val1 = denom * gradient
        val2 = np.ones_like(self.data) * (np.sum(gradient * self.data))
        self.append_gradient((val1 - val2)/(denom*denom))
    return t.TensorNode(forward_update(), False, set([self]), forward_update, update_policy) 

def sum(self: t.TensorNode, axis: tuple,keepDim=False) -> t.TensorNode:
    """
    Performs the numpy operation of np.sum. 
    Args:
        self (TensorNode): the tensor. P
        axis (tuple): The axes to sum over
        keepDim (boolean): optional variable determining of dimensions are kept. Default is False
    Returns:
         the tensor with data np.sum(data,axis=axis=,keepdims=keepDim)
    """
    self.out_degree += 1
    def forward_update():
        return np.sum(self.data,axis=axis,keepdims=keepDim)
    def update_policy(gradient: np.ndarray):
        self.append_gradient(gradient) # looks like cheating but it duplicates the gradient like a sum should
    return t.TensorNode(forward_update(),False,set([self]),forward_update,update_policy)

#dp/dt = kp(1-p/L) = p(1-p) as k = 1 and L = 1
def sigmoid(self: t.TensorNode) -> t.TensorNode:
    self.out_degree += 1
    def forward_update():
        return 1/(1+np.exp(-self.data)) # i love numpy broadcasting
    def update_policy(gradient: np.ndarray):
        dx = forward_update()*(1-forward_update())
        self.append_gradient(gradient * dx)
    return t.TensorNode(forward_update(),False,set([self]),forward_update,update_policy)