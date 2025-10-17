import numpy as np
from autograd.tensor import Tensor
from autograd.operations.conv_2d_operation import Conv2DOperation, FlattenOperation
from autograd.operations.max_pool_2d_operation import MaxPool2dOperation

class SimpleCNN:
    def __init__(self, K: int = 4, num_classes: int = 10):
        self.K = K
        self.num_classes = num_classes
        self.kernels = [Tensor(np.random.randn(3, 3) * 0.1, requires_grad=True) for _ in range(K)]
        self.W = [Tensor(np.random.randn(num_classes, 169) * 0.1, requires_grad=True) for _ in range(K)]
        self.b = [Tensor(np.zeros((num_classes, 1), dtype=np.float32), requires_grad=True) for _ in range(K)]

    def __call__(self, image: Tensor):
        return self.forward(image)

    def forward(self, image: Tensor):
        logits = None
        for k in range(self.K):
            x = Conv2DOperation.apply(image, self.kernels[k]).relu()
            x = MaxPool2dOperation.apply(x, 2) 
            x = FlattenOperation.apply(x)
            branch = self.W[k] @ x + self.b[k] 
            logits = branch if logits is None else (logits + branch)
        return logits

    def parameters(self):
        return [*self.kernels, *self.W, *self.b]