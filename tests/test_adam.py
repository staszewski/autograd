import numpy as np
from autograd.tensor import Tensor
from autograd.adam import Adam

def _scalar(x):
    return Tensor(np.array([[x]], dtype=np.float32), requires_grad=True)

def _loss_quadratic(w):
    return 0.5 * (w * w)

def test_adam_first_step_bias_correction():
    p = _scalar(0.0)
    opt = Adam([p], lr=0.1, betas=(0.9, 0.999), eps=1e-8)
    # Simulate grad=1 for first step
    p._grad = np.array([[1.0]], dtype=np.float32)
    opt.step()
    # After 1st step with constant grad=1:
    # m1=(1-0.9)*1=0.1, v1=(1-0.999)*1=0.001
    # m_hat=0.1/0.1=1, v_hat=0.001/0.001=1 → Δ=lr*1/sqrt(1)=lr
    assert np.allclose(p.data, -0.1, atol=1e-8)

def test_adam_converges_on_quadratic():
    w = _scalar(3.0)
    lr = 0.1
    opt = Adam([w], lr=lr, betas=(0.9, 0.999), eps=1e-8)
    for _ in range(200):
        loss = _loss_quadratic(w)
        w.zero_grad()
        loss.backward()
        opt.step()

    assert np.abs(w.data).item() < 1e-2