import Tensor as t
import numpy as np
import utility as util
#TODO: update all of these functions to utilize the thing
def dot(self: t.TensorNode,other: t.TensorNode) -> t.TensorNode:
    """
    Computes the generalized dot product between the two Tensors
    Args:
        self (TensorNode): first tensor
        other (TensorNode): second tensor
    Returns:
        The generalized dot product self dot other
    """
    access = util.find_out_dependents(self,other)
    def update_policy(gradient: np.ndarray):
        if(isinstance(self,t.TensorNode)):
            self.append_gradient(gradient * access[2])
        if(isinstance(other,t.TensorNode)):
            other.append_gradient(gradient * access[1])
    def forward_update():
        return np.sum(access[1] * access[2])
    return t.TensorNode(forward_update(), access[0], forward_update, update_policy)
    
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
    return t.TensorNode(forward_update(), set([self]), forward_update, update_policy)

def log(self: t.TensorNode) -> t.TensorNode:
    """
    Activation function that takes the natural log of each element.
    Args:
        self (TensorNode): the tensor
        Returns:
            TensorNode representing log(self)
    """
    self.out_degree += 1
    def update_policy(gradient: np.ndarray):
        self.append_gradient(gradient/self.data)
    def forward_update():
        return np.log(self.data)
    return t.TensorNode(forward_update(), set([self]), forward_update, update_policy)

def softmaxCrossEntropy(self: t.TensorNode, other: t.TensorNode, axis: int):
    """
    Returns an efficent and numerically stable version of the softmax of self AND cross entropy of other with respect to self.
    Args:
        self (TensorNode): the tensorNode model
        other (np.ndarray): the model's true values. Please ensure that np.sum(other,axis=axis) = 1 i.e. that the axis is a valid probability distribution.
        axis (int): axis by which to sum values for softmax.
    Returns:
        TensorNode representing the cross entropy of the softmax.
    """
    self.out_degree+=1
    if(isinstance(other,t.TensorNode)):
        other = other.data
    def forward_update():
        maximal = np.max(self.data,axis=axis, keepdims=True)
        exponential = np.exp(self.data - maximal)
        summy = np.sum(exponential,axis=axis, keepdims=True)
        return -np.sum(other * (self.data-maximal-np.log(summy)))
    def update_policy(gradient: np.ndarray):
        maximal = np.max(self.data,axis=axis, keepdims=True)
        exponential = np.exp(self.data - maximal)
        summy = np.sum(exponential,axis=axis, keepdims=True)
        self.append_gradient(gradient*(exponential/summy - other))
    return t.TensorNode(forward_update(),set([self]),forward_update,update_policy)

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
    return t.TensorNode(forward_update(), set([self]), forward_update, update_policy)


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
    return t.TensorNode(forward_update(), set([self]), forward_update, update_policy)

def exp(self: t.TensorNode) -> t.TensorNode:
    """
    Performs element wise exponent stuff
    """
    def forward_update():
        return np.exp(self.data)
    def update_policy(gradient: np.ndarray):
        self.append_gradient(gradient * forward_update())
    return t.TensorNode(forward_update(),set([self]), forward_update, update_policy)


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
    return t.TensorNode(forward_update(), set([self]), forward_update, update_policy) 

def sum(self: t.TensorNode, axis: tuple) -> t.TensorNode:
    """
    Performs the numpy operation of np.sum. 
    Args:
        self (TensorNode): the tensor. P
        axis (tuple): The axes to sum over
    Returns:
         the tensor with data np.sum(data,axis=axis=,keepdims=keepDim)
    """
    self.out_degree += 1
    def forward_update():
        return np.sum(self.data,axis=axis,keepdims=True)
    def update_policy(gradient: np.ndarray):
        self.append_gradient(gradient) # looks like cheating but it duplicates the gradient like a sum should
    return t.TensorNode(forward_update(),set([self]),forward_update,update_policy)

#dp/dt = kp(1-p/L) = p(1-p) as k = 1 and L = 1
def sigmoid(self: t.TensorNode) -> t.TensorNode:
    self.out_degree += 1
    def forward_update():
        return 1/(1+np.exp(-self.data)) # i love numpy broadcasting
    def update_policy(gradient: np.ndarray):
        dx = forward_update()*(1-forward_update())
        self.append_gradient(gradient * dx)
    return t.TensorNode(forward_update(),set([self]),forward_update,update_policy)

def tanh(self: t.TensorNode) -> t.TensorNode:
    self.out_degree += 1
    def forward_update():
        return np.tanh(self.data)
    def update_policy(gradient):
        self.append_gradient(-gradient/np.cosh(self.data)**2)
    return t.TensorNode(forward_update(),set([self]),forward_update,update_policy)

