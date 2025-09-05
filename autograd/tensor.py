import numpy as np
from typing import Optional, Set, Callable, Union

class Tensor:
    def __init__(self, data, requires_grad=False):
        # TODO: What should we store here?
        self._data = np.array(data)
        self._requires_grad = requires_grad
        self._grad = np.zeros_like(self._data)
        self._prev: Set[Tensor] = set()
        self._backward_fn: Optional[Callable] = None
        pass

    def __repr__(self):
        """String representation of the tensor."""
        return f"Tensor({self._data}, requires_grad={self._requires_grad})"

    @property
    def data(self):
        """Access the underlying data."""
        return self._data

    @property
    def requires_grad(self):
        """Check if gradients are required."""
        return self._requires_grad

    def backward(self, grad=None):
        """Compute the gradient of the tensor."""
        if self._requires_grad:
            if grad is None:
                grad = np.ones_like(self._data)

            self._grad += grad

            if self._backward_fn is not None:
                self._backward_fn()

        else:
            raise RuntimeError("Gradient computation is not allowed for this tensor.")

    def __add__(self, other):
        """Addition operation with autograd support."""
        # Handle the case where other might be a number
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        # Forward pass: compute the result
        result_data = self._data + other._data
        result = Tensor(result_data, requires_grad=self._requires_grad or other._requires_grad)
        
        # Set up computational graph
        result._prev = {self, other} 
        
        # Define backward function
        def backward_fn():
            if self._requires_grad:
                self.backward(result._grad)
            if other._requires_grad:
                other.backward(result._grad)
        
        result._backward_fn = backward_fn
        return result
