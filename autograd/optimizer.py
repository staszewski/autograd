from typing import Iterable, List
from autograd.tensor import Tensor
import numpy as np

class Optimizer:
    def __init__(self, params: Iterable[Tensor]) -> None:
        flat: List[Tensor] = []
        for p in params:
            if isinstance(p, (list, tuple)):
                for q in p:
                    if getattr(q, "_requires_grad", False):
                        flat.append(q)
            else:
                if getattr(p, "_requires_grad", False):
                    flat.append(p)
        self.params: List[Tensor] = flat

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        raise NotImplementedError("To be overwritten in a subclass")

class SGD(Optimizer):
    def __init__(self, params: Iterable[Tensor], lr: float = 1e-2, momentum: float = 0.0) -> None:
        super().__init__(params)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self._velocity = {id(p): np.zeros_like(p._data) for p in self.params}

    def step(self):
        for p in self.params:
            g = p._grad
            if self.momentum > 0.0:
                v = self._velocity[id(p)] = self.momentum * self._velocity[id(p)] + g
                p._data = p._data - self.lr * v
            else:
                p._data = p._data - self.lr * p._grad