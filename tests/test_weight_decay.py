import numpy as np
from autograd.tensor import Tensor
from autograd.optimizer import SGD
from autograd.adam import Adam

def _scalar(x): return Tensor(np.array([[x]], dtype=np.float32), requires_grad=True)

def test_sgd_weight_decay_decoupled():
    w = _scalar(1.0)
    w._grad = np.array([[0.0]], dtype=np.float32)
    opt = SGD([w], lr=0.1, weight_decay=0.01)
    opt.step()
    # decoupled: w <- w - lr*wd*w = (1 - lr*wd) w
    assert np.allclose(w.data, 1.0 * (1 - 0.1*0.01))

def test_adam_weight_decay_decoupled():
    w = _scalar(1.0); w._grad = np.array([[0.0]], dtype=np.float32)
    opt = Adam([w], lr=0.1, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01)
    opt.step()
    assert np.allclose(w.data, 1.0 * (1 - 0.1*0.01))