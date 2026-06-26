#imports
import numpy as np

#sample 2d tensor (technically 2d due to double array)
x = np.array([[1,2,3]])

#[1 2 3] -- prints in lin alg notation
print(x)

print(type(x))

# (1, 3)
print(x.shape)

y = np.array([[1],[2],[3]])

# (3, 1)
print(y.shape)

# "@" operator is matmul for 2d tensors

#sample inner product (1,3) x (3,1) = (1,1)
print(x @ y)

#sample outer product (3,1) x (1,3) = (3,3)

print(y @ x)

#transpose
z = x.T
print(z)


