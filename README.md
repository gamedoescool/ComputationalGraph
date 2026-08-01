**This is my implementation of the dynamic computational graph and autograd using a custom built TensorNode object. Math is also explained**

Purpose is to sharpen my skills and get a deeper understanding of how AI scales

*files*
**Matrix Calculus.ipynb**: Explains Matrix Calculus, the math that makes backprop efficient. Assumes Linear Algebra and knowledge of gradients, partial derivatives, and the jacobian. Highly recommended to start here.

**Theory.ipynb**: the colab notebook explaining the overall theory. Start here if you are comfortable with Matrix Calculus.

**backprop_proofs.ipynb**: File contains the proofs for how to update the gradient of common Tensor operations. 

**Tensor.py**: Contians the TensorNode class for the computational graph

**Compile.py**: Class that navigates the topo sort list. 

**utility.py**: Utility class implementing various special functions

**How to run**

First, you have to install all dependencies using `pip install -r requirements`.

Then you can run MNIST_test.ipynb to see how it does on the MNIST dataset. 

