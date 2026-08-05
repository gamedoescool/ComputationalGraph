import numpy as np
import Tensor as t
def condense(input: np.ndarray, shape: tuple):
    """
    Condenses the input into a given shape. Uses np.sum to condense axes
    Args:
        input (np.ndarray): input to be condensed.
        shape (tuple): the final shape input is to be condensed into
    Returns:
        np.ndarray representing condensed input"""
    #pass 1: trim dimensions
    reduce = []
    for j in range(len(shape),len(input.shape)):
        reduce.append(j)

    temp = np.sum(input,axis=tuple(reduce))

    #pass 2: condense  1's
    reduce = []
    for j in range(len(shape)):
        if(temp.shape[j] != shape[j]):
            reduce.append(j)
    return np.sum(temp,axis=tuple(reduce), keepdims=True)

def sAct(data: np.ndarray):
    """
    Implements my custom activation function. A smooth strictly positive function that behaves similarly to RELU.
    """
    mask = (data >= 0)
    mask1 = (data < 0)
    a = np.zeros_like(data)
    a[mask] = data[mask]+1
    a[mask1] = np.exp(data[mask1])
    return a

def sActPrime(data: np.ndarray):
    """
    Implements the derivative of my custom activation function. A smooth strictly positive function that behaves similarly to RELU.
    """
    mask = (data >= 0)
    mask1 = (data < 0)
    a = np.zeros_like(data)
    a[mask] = 1
    a[mask1] = np.exp(data[mask1])
    return a

def find_out_dependents(self, other):
    """
    Technical functoin that returns dependents and where to access values
    Args:
        self: tensor or raw data
        other: tensor or raw data
    output:
        [dependents, selfData,otherData]"""

    if(not isinstance(self,t.TensorNode)):
        other.out_degree += 1
        return [set([other]),self]
    if(not isinstance(other,t.TensorNode)):
        self.out_degree += 1
        return [set([self]),self.data,other]
    
    other.out_degree += 1
    self.out_degree += 1
    return [set([self]),self.data,other]
