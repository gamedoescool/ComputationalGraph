**This is my implementation of the dynamic computational graph and autograd using a custom built TensorNode object. Math is also explained**

No AI, agentic agent, or LLM's was used to create this, as the purpose of this project was to sharpen my skills.

Dependencies:

numpy

*files*
**Matrix Calculus.ipynb**: Explains Matrix Calculus, the math that makes backprop efficient. Assumes Linear Algebra and knowledge of gradients, partial derivatives, and the jacobian. Highly recommended to start here.

**Theory.ipynb**: the colab notebook explaining the overall theory. Start here if you are comfortable with Matrix Calculus.

**Test.py**: File that tests basic numpy behavior. Mostly for me to get accustomed to how numpy treates ndarrays

**backprop_proofs.ipynb**: File contains the proofs for how to update the gradient of common Tensor operations. 

**How to run**

First, you have to install all dependencies using pip install requirements.txt