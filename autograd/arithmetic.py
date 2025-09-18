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
        a, b = ctx.saved_tensors

        def unbroadcast(grad, shape):
            ndims_added = grad.ndim - len(shape)
            if ndims_added > 0:
                grad = grad.sum(axis=tuple(range(ndims_added)))
        
            for axis, size in enumerate(shape):
                if size == 1 and grad.shape[axis] > 1:
                    grad = grad.sum(axis=axis, keepdims=True)
        
            return grad
        return [unbroadcast(grad_output, a.data.shape),
            unbroadcast(grad_output, b.data.shape)]

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

class MatMulOperation(Operation):
    """Matrix multiplication operation."""
    
    @classmethod
    def forward(cls, ctx: Context, a: 'Tensor', b: 'Tensor') -> np.ndarray:
        ctx.save_for_backward(a, b)
        return a.data @ b.data
    
    @classmethod
    def backward(cls, ctx: Context, grad_output: np.ndarray) -> List[np.ndarray]:
        a, b = ctx.saved_tensors

        return [grad_output @ b.data.T, a.data.T @ grad_output]

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


class DivOperation(Operation):
    """Division operation."""
    
    @classmethod
    def forward(cls, ctx: Context, a: 'Tensor', b: 'Tensor') -> np.ndarray:
        ctx.save_for_backward(a, b)
        return a.data / b.data
    
    @classmethod
    def backward(cls, ctx: Context, grad_output: np.ndarray) -> List[np.ndarray]:
        a, b = ctx.saved_tensors

        # quotient rule h'(x) = [f'(x) × g(x) - f(x) × g'(x)] / [g(x)]²
        return [grad_output / b.data, -grad_output * a.data / (b.data ** 2)]

class ReLUOperation(Operation):
    """ReLU activation function operation."""
    
    @classmethod
    def forward(cls, ctx: Context, a: 'Tensor') -> np.ndarray:
        ctx.save_for_backward(a)

        """
        ReLU forward: 0 if x < 0, x if x >= 0
        """
        return np.maximum(0, a.data)
    
    @classmethod
    def backward(cls, ctx: Context, grad_output: np.ndarray) -> List[np.ndarray]:
        a, = ctx.saved_tensors

        """
        ReLU derivative: 1 if x > 0, else 0
        """
        grad_a = grad_output * (a.data > 0)
        return [grad_a]

class SigmoidOperation(Operation):
    """Sigmoid activation function operation."""
    
    @classmethod
    def forward(cls, ctx: Context, a: 'Tensor') -> np.ndarray:
        sigmoid_output = 1 / (1 + np.exp(-a.data))
        ctx.save_for_backward(a)
        ctx.save_for_backward_values(sigmoid_output)
        return sigmoid_output
    
    @classmethod
    def backward(cls, ctx: Context, grad_output: np.ndarray) -> List[np.ndarray]:
        sigmoid_output, = ctx.saved_values

        grad_a = grad_output * sigmoid_output * (1 - sigmoid_output) 
        return [grad_a]

class TanhOperation(Operation):
    """Tanh activation function operation."""
    
    @classmethod
    def forward(cls, ctx: Context, a: 'Tensor') -> np.ndarray:
        tanh_output = np.tanh(a.data)
        ctx.save_for_backward(a)
        ctx.save_for_backward_values(tanh_output)
        return tanh_output
    
    @classmethod
    def backward(cls, ctx: Context, grad_output: np.ndarray) -> List[np.ndarray]:
        tanh_output, = ctx.saved_values

        # tanh'(x) = 1 - tanh²(x)
        grad_a = grad_output * (1 - tanh_output ** 2)
        return [grad_a]