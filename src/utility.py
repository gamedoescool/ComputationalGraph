import numpy as np
def condense(input: np.ndarray, shape: tuple, out):
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
    np.sum(temp,axis=tuple(reduce), keepdims=True,out=out)

def sAct(data: np.ndarray):
    mask = (data >= 0)
    mask1 = (data < 0)
    a = np.zeros_like(data)
    a[mask] = data[mask]+1
    a[mask1] = np.exp(data[mask1])
    return a

def sActPrime(data: np.ndarray):
    mask = (data >= 0)
    mask1 = (data < 0)
    a = np.zeros_like(data)
    a[mask] = 1
    a[mask1] = np.exp(data[mask1])
    return a