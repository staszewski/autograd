from abc import ABC, abstractmethod
import numpy as np

from autograd.context import Context
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from autograd.tensor import Tensor

class Operation(ABC):
    """Base class for all operations."""
    
    @abstractmethod
    def forward(self, *inputs: 'Tensor') -> np.ndarray:
        """Forward pass."""
        pass
    
    @abstractmethod
    def backward(self, *grads: 'Tensor') -> List[np.ndarray]:
        """Backward pass."""
        pass

    @classmethod
    def apply(cls, *args: 'Tensor') -> 'Tensor':
        """Apply the operation to the given inputs."""
        # Import here to avoid circular import
        from autograd.tensor import Tensor

        ctx = Context()
        output = cls.forward(ctx, *args)

        requires_grad = any(arg.requires_grad for arg in args)

        result = Tensor(output.data, requires_grad=requires_grad)
        result._grad_fn = (cls, ctx)
        result._prev = set(args)

        return result