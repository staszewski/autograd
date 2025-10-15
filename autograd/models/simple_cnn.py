import numpy as np
from autograd.tensor import Tensor
from autograd.operations.conv_2d_operation import Conv2DOperation, FlattenOperation
from autograd.operations.max_pool_2d_operation import MaxPool2dOperation

class SimpleCNN:
    def __init__(self):
        self.conv1_kernel = Tensor(np.random.randn(3, 3) * 0.1, requires_grad=True)
        self.W = Tensor(np.random.randn(10, 169) * 0.1, requires_grad=True)
        self.b = Tensor(np.zeros((10, 1), dtype=np.float32), requires_grad=True)

    def __call__(self, image):
        return self.forward(image)

    def forward(self, image):
        x = Conv2DOperation.apply(image, self.conv1_kernel).relu()
        x = MaxPool2dOperation.apply(x, 2)
        x = FlattenOperation.apply(x)            # (169, 1)
        logits = self.W @ x + self.b            # (10, 1)
        return logits

    def parameters(self):
        return [self.conv1_kernel, self.W, self.b]