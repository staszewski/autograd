from autograd.context import Context
from autograd.operation import Operation

import math
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

class MulOperation(Operation):
    """Multiplication operation."""
    
    @classmethod
    def forward(cls, ctx: Context, a: 'Tensor', b: 'Tensor') -> np.ndarray:
        ctx.save_for_backward(a, b)
        return a.data * b.data
    
    @classmethod
    def backward(cls, ctx: Context, grad_output: np.ndarray) -> List[np.ndarray]:
        a, b = ctx.saved_tensors

        return [grad_output * b.data, grad_output * a.data]

class PowOperation(Operation):
    """Power operation."""
    
    @classmethod
    def forward(cls, ctx: Context, a: 'Tensor', n: 'Tensor') -> np.ndarray:
        ctx.save_for_backward(a, n)
        return a.data ** n.data
    
    @classmethod
    def backward(cls, ctx: Context, grad_output: np.ndarray) -> List[np.ndarray]:
        a, n = ctx.saved_tensors

        # d/da(a^n) = n * a^(n-1) -> Power Rule
        grad_a = grad_output * n.data * (a.data ** (n.data - 1))

        eps = 1e-8  # Small positive number

        if a.data <= 0:
            log_a = math.log(eps)
        else:
            log_a = math.log(a.data)

        # d/dn(a^n) = a^n * ln(a) -> Exponential Rule
        grad_n = grad_output * (a.data ** n.data) * log_a 

        return [grad_a, grad_n]
