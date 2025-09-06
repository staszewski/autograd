import numpy as np
from typing import Optional, Set, Callable, Union

class Tensor:
    def __init__(self, data, requires_grad=False):
        self._data = np.array(data)
        self._requires_grad = requires_grad
        self._grad = np.zeros_like(self._data)
        self._prev: Set[Tensor] = set()
        self._backward_fn: Optional[Callable] = None
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

            if self._backward_fn is not None:
                self._backward_fn()

        if is_root_call:
            delattr(Tensor, '_backward_visited')
            
        if not self._requires_grad:
            raise RuntimeError("Gradient computation is not allowed for this tensor.")

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        result_data = self._data + other._data
        result = Tensor(result_data, requires_grad=self._requires_grad or other._requires_grad)
        
        result._prev = {self, other} 
        
        def backward_fn():
            if self._requires_grad:
                self.backward(result._grad)
            if other._requires_grad:
                other.backward(result._grad)
        
        result._backward_fn = backward_fn
        return result

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        result_data = self._data - other._data
        result = Tensor(result_data, requires_grad=self._requires_grad or other._requires_grad)
        result._prev = {self, other}

        def backward_fn():
            if self._requires_grad:
                self.backward(result._grad)
            if other._requires_grad:
                other.backward(-result._grad)
        
        result._backward_fn = backward_fn
        return result