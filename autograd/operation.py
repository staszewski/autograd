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
    def apply(cls, *args: 'Tensor', **kwargs) -> 'Tensor':
        """Apply the operation to the given inputs."""
        # Import here to avoid circular import
        from autograd.tensor import Tensor

        ctx = Context()
        output = cls.forward(ctx, *args, **kwargs)
        
        tensor_args = [a for a in args if isinstance(a, Tensor)]
        tensor_kwargs = [v for v in kwargs.values() if isinstance(v, Tensor)] 
        tensors = tensor_args + tensor_kwargs

        requires_grad = any(t.requires_grad for t in tensors)

        result = Tensor(output.data, requires_grad=requires_grad)
        result._grad_fn = (cls, ctx)
        result._prev = set(tensors)

        return result