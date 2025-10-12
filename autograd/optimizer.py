from typing import Iterable, List
from autograd.tensor import Tensor

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
    def __init__(self, params: Iterable[Tensor], lr: float = 1e-2) -> None:
        super().__init__(params)
        self.lr = float(lr)

    def step(self):
        for p in self.params:
            p._data = p._data - self.lr * p._grad