from autograd.context import Context
from autograd.operation import Operation

import numpy as np
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from autograd.tensor import Tensor


class AddOperation(Operation):
    """Addition operation."""
    
    @classmethod
    def forward(cls, ctx: Context, a: 'Tensor', b: 'Tensor') -> np.ndarray:
        ctx.save_for_backward(a, b)
        return a.data + b.data
    
    @classmethod
    def backward(cls, ctx: Context, grad_output: np.ndarray) -> List[np.ndarray]:
        return [grad_output, grad_output]

class SubOperation(Operation):
    """Subtraction operation."""
    
    @classmethod
    def forward(cls, ctx: Context, a: 'Tensor', b: 'Tensor') -> np.ndarray:
        ctx.save_for_backward(a, b)
        return a.data - b.data
    
    @classmethod
    def backward(cls, ctx: Context, grad_output: np.ndarray) -> List[np.ndarray]:
        return [grad_output, -grad_output]