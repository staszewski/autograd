import numpy as np
from typing import Optional, Set, Tuple, List, Any

from autograd.arithmetic import AddOperation, MulOperation, SubOperation
from autograd.context import Context

class Tensor:
    def __init__(self, data, requires_grad=False):
        self._data = np.array(data)
        self._requires_grad = requires_grad
        self._grad = np.zeros_like(self._data)
        self._prev: Set[Tensor] = set()

        self._grad_fn: Optional[Tuple[type, Context]] = None
        pass

    def __repr__(self):
        return f"Tensor({self._data}, requires_grad={self._requires_grad})"

    @property
    def data(self):
        return self._data

    @property
    def requires_grad(self):
        return self._requires_grad

    def backward(self, grad=None):
        is_root_call = grad is None
        
        if is_root_call:
            Tensor._backward_visited = set()
        
        if self._requires_grad:
            if grad is None:
                grad = np.ones_like(self._data)

            self._grad += grad

            if self in Tensor._backward_visited:
                return
            Tensor._backward_visited.add(self)

            if self._grad_fn is not None:
                op_class, ctx = self._grad_fn
                grad_inputs = op_class.backward(ctx, self._grad)
                
                for i, grad_input in enumerate(grad_inputs):
                    input_tensor = ctx.saved_tensors[i]
                    if input_tensor._requires_grad:
                        input_tensor.backward(grad_input)

        if is_root_call:
            delattr(Tensor, '_backward_visited')
            
        if not self._requires_grad:
            raise RuntimeError("Gradient computation is not allowed for this tensor.")

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

def _ensure_tensor(value):
    """Ensure value is a Tensor."""
    if isinstance(value, Tensor):
        return value
    return Tensor(value)