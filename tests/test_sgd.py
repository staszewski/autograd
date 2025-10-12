import numpy as np
from autograd.tensor import Tensor
from autograd.optimizer import SGD

def _scalar(x):
    return Tensor(np.array([[x]], dtype=np.float32), requires_grad=True)

def _loss_quadratic(w):
    # L(w) = 0.5 * w^2  (scalar)
    return 0.5 * (w * w)

def test_sgd_one_step_matches_rule():
    w0 = _scalar(3.0)
    opt = SGD([w0], lr=0.1) 

    loss = _loss_quadratic(w0)
    loss.backward()

    # w1 = w0 - lr * grad = 3 - 0.1 * 3 = 2.7
    opt.step()
    assert np.allclose(w0.data, 2.7, atol=1e-6)

def test_sgd_descends_and_converges_on_quadratic():
    w = _scalar(3.0)
    lr = 0.2
    opt = SGD([w], lr=lr)

    last = None
    for _ in range(30):
        loss = _loss_quadratic(w)
        if last is not None:
            assert loss.data <= last + 1e-8 
        w.zero_grad()
        loss.backward()
        opt.step()
        last = loss.data

    assert np.abs(w.data).item() < 1e-2
