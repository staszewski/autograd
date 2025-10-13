import numpy as np
from autograd.tensor import Tensor
from autograd.optimizer import SGD

def _scalar(x):
    return Tensor(np.array([[x]], dtype=np.float32), requires_grad=True)

def _loss_quadratic(w):
    # L(w) = 0.5 * w^2 
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

def test_sgd_momentum():
    w_van = _scalar(3.0)
    w_mom = _scalar(3.0)

    van = SGD([w_van], lr=0.1)
    mom = SGD([w_mom], lr=0.1, momentum=0.9)

    def run(w, opt, tol=1e-2, max_steps=200):
        for t in range(1, max_steps + 1):
            loss = _loss_quadratic(w)
            w.zero_grad()
            loss.backward()
            opt.step()
            if np.abs(w.data).item() < tol:
                return t
        return max_steps + 1

    steps_v = run(w_van, van)
    steps_m = run(w_mom, mom)
    assert steps_m < steps_v