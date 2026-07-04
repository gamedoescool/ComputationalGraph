import numpy as np
#wait i think... this is it?
def condense(input: np.ndarray, shape: tuple, out):
    #pass 1: trim dimensions
    reduce = []
    for j in range(len(shape),len(input.shape)):
        reduce.append(j)
    np.sum(input,axis=tuple(reduce),out=out)

    #pass 2: condense  1's
    reduce = []
    for j in range(len(shape)):
        if(input.shape[j] != shape[j]):
            reduce.append(j)
    np.sum(input,axis=tuple(reduce), keepdims=True,out=out)
    
    
z = np.random.uniform(0,1,size=(4,3))
v = np.random.uniform(0,1,size=(4,3))
k = np.zeros_like(v)
print(z.shape) #(4,3,1,69)
z = condense(z,v.shape,out=v)
print(v.shape) # (4,3)
