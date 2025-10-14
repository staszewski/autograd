from autograd.optimizer import Optimizer
from typing import Iterable, List, Tuple
from autograd.tensor import Tensor

import numpy as np

class Adam(Optimizer):
    def __init__(self, params: Iterable[Tensor], lr: float = 1e-2, betas: Tuple[float, float]=(0.0, 0.0), eps: float = 1e-8, weight_decay: float = 0.0) -> None:
        super().__init__(params)
        self.lr = float(lr)
        self.beta1, self.beta2 = betas
        self.eps = float(eps)
        self._m = {id(p): np.zeros_like(p._data) for p in self.params}
        self._v = {id(p): np.zeros_like(p._data) for p in self.params}
        self._t = 0
        self.weight_decay = weight_decay

    def step(self, grad_clip: float | None = None):
        self._clip_grads(grad_clip)
        self._t += 1
        b1t = 1.0 - self.beta1 ** self._t
        b2t = 1.0 - self.beta2 ** self._t
        for p in self.params:
            g = p._grad
            mid = id(p)
            m = self._m[mid] = self.beta1 * self._m[mid] + (1.0 - self.beta1) * g
            v = self._v[mid] = self.beta2 * self._v[mid] + (1.0 - self.beta2) * (g * g)
            m_hat = m / b1t
            v_hat = v / b2t

            if self.weight_decay > 0.0:
                p._data -= self.lr * self.weight_decay * p._data

            p._data = p._data - self.lr * (m_hat / (np.sqrt(v_hat) + self.eps))  
                