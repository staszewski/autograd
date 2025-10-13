import numpy as np
import pytest
from autograd.tensor import Tensor
from autograd.optimizer import Optimizer

def test_optimizer_filters_only_trainable_params():
    p1 = Tensor(np.ones((2,2)), requires_grad=True)
    p2 = Tensor(np.ones((2,2)), requires_grad=False)
    opt = Optimizer([p1, p2])  # should keep only p1
    assert len(opt.params) == 1
    assert opt.params[0] is p1

def test_zero_grad_resets_all_param_grads():
    p = Tensor(np.ones((3,)), requires_grad=True)
    p._grad = np.ones_like(p._data)
    opt = Optimizer([p])
    opt.zero_grad()
    assert np.allclose(p._grad, 0.0)

def test_step_is_abstract():
    p = Tensor(np.ones((1,)), requires_grad=True)
    opt = Optimizer([p])
    with pytest.raises(NotImplementedError):
        opt.step()