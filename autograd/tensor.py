import numpy as np
from typing import Optional, Set, Tuple

from autograd.arithmetic import AddOperation, MatMulOperation, MulOperation, PowOperation, ReLUOperation, SigmoidOperation, SubOperation, DivOperation, TanhOperation
from autograd.context import Context

class Tensor:
    def __init__(self, data, requires_grad=False):
        self._data = np.array(data)
        if requires_grad and not np.issubdtype(self._data.dtype, np.floating):
            self._data = self._data.astype(np.float32)
        self._requires_grad = requires_grad
        self._grad = np.zeros_like(self._data)
        self._prev: Set[Tensor] = set()

        self._grad_fn: Optional[Tuple[type, Context]] = None
        pass

    def __repr__(self):
        return f"Tensor({self._data}, requires_grad={self._requires_grad})"

    def zero_grad(self):
        self._grad = np.zeros_like(self._data)

    @property
    def data(self):
        return self._data

    @property
    def requires_grad(self):
        return self._requires_grad

    def backward(self, grad=None):
        """
        Compute gradients using backpropagation.
        
        Args:
            grad: Optional gradient to start with. If None, uses ones_like(self._data)
        """
        if not self._requires_grad:
            raise RuntimeError("Gradient computation is not allowed for this tensor.")
        
        if grad is None:
            grad = np.ones_like(self._data)
        
        visited = set()
        self._backward_impl(grad, visited)
    
    def _backward_impl(self, grad, visited):
        """
        Internal backward implementation with explicit visited tracking.
        
        Args:
            grad: The gradient flowing into this tensor
            visited: Set of tensors already visited in this backward pass
        """
        self._grad += grad
        
        if self in visited:
            return
        visited.add(self)
        
        if self._grad_fn is not None:
            op_class, ctx = self._grad_fn
            grad_inputs = op_class.backward(ctx, self._grad)
            
            for i, grad_input in enumerate(grad_inputs):
                input_tensor = ctx.saved_tensors[i]
                if input_tensor._requires_grad:
                    input_tensor._backward_impl(grad_input, visited)

    def relu(self):
        return ReLUOperation.apply(self)

    def sigmoid(self):
        return SigmoidOperation.apply(self)

    def tanh(self):
        return TanhOperation.apply(self)

    def __neg__(self):
        return self * -1

    def __pos__(self):
        return self

    def __add__(self, other):
        return AddOperation.apply(self, _ensure_tensor(other))

    def __radd__(self, other):
        return AddOperation.apply(_ensure_tensor(other), self)

    def __sub__(self, other):
        return SubOperation.apply(self, _ensure_tensor(other))

    def __rsub__(self, other):
        return SubOperation.apply(_ensure_tensor(other), self)

    def __mul__(self, other):
        return MulOperation.apply(self, _ensure_tensor(other))

    def __rmul__(self, other):
        return MulOperation.apply(_ensure_tensor(other), self)

    def __matmul__(self, other):
        return MatMulOperation.apply(self, _ensure_tensor(other))

    def __truediv__(self, other):
        return DivOperation.apply(self, _ensure_tensor(other))

    def __rtruediv__(self, other):
        return DivOperation.apply(_ensure_tensor(other), self)

    def __pow__(self, other):
        return PowOperation.apply(self, _ensure_tensor(other))

    def __rpow__(self, other):
        return PowOperation.apply(_ensure_tensor(other), self)

def _ensure_tensor(value):
    """Ensure value is a Tensor."""
    if isinstance(value, Tensor):
        return value
    return Tensor(value)