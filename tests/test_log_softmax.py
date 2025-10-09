import numpy as np
import pytest
from autograd.tensor import Tensor
from autograd.operations.log_softmax import LogSoftmaxOperation
from autograd.operations.softmax import SoftmaxOperation

def test_log_softmax_matches_log_of_softmax_on_moderate_logits_axis0():
    logits = Tensor([[1.0, -2.0],
                     [0.0,  3.0],
                     [2.0,  1.0]])
    ls = LogSoftmaxOperation.apply(logits, axis=0).data
    sm = SoftmaxOperation.apply(logits, axis=0).data
    assert np.allclose(ls, np.log(sm), atol=1e-7)

def test_log_softmax_matches_log_of_softmax_on_moderate_logits_axis1():
    logits = Tensor([[1.0, -2.0, 0.5]])
    ls = LogSoftmaxOperation.apply(logits, axis=1).data
    sm = SoftmaxOperation.apply(logits, axis=1).data
    assert np.allclose(ls, np.log(sm), atol=1e-7)

def test_log_softmax_is_stable_on_extreme_logits():
    big = 1e3
    logits0 = Tensor([[big, -big, 0.0]]).data.T
    logits1 = Tensor([[big, -big, 0.0]])
    ls0 = LogSoftmaxOperation.apply(logits0, axis=0).data
    ls1 = LogSoftmaxOperation.apply(logits1, axis=1).data
    assert np.all(np.isfinite(ls0))
    assert np.all(np.isfinite(ls1))
    # exp(log_softmax) still sums to 1 along the axis
    assert np.allclose(np.exp(ls0).sum(axis=0, keepdims=True), 1.0, atol=1e-7)
    assert np.allclose(np.exp(ls1).sum(axis=1, keepdims=True), 1.0, atol=1e-7)


@pytest.mark.parametrize("axis", [0, 1])
def test_log_softmax_backward_finite_difference(axis):
    rng = np.random.default_rng(42)
    dim = 4
    logits_data = rng.standard_normal((dim, 1)) if axis == 0 else rng.standard_normal((1, dim))
    logits = Tensor(logits_data, requires_grad=True)

    log_probs = LogSoftmaxOperation.apply(logits, axis=axis)
    upstream = rng.standard_normal(size=log_probs.data.shape)
    log_probs.backward(upstream)

    eps = 1e-6
    fd = np.zeros_like(logits.data)
    for idx in np.ndindex(logits_data.shape):
        # perturb
        pert_plus = logits_data.copy()
        pert_minus = logits_data.copy()
        pert_plus[idx] += eps
        pert_minus[idx] -= eps
        # recompute probs
        lp_plus = LogSoftmaxOperation.apply(Tensor(pert_plus), axis=axis).data
        lp_minus = LogSoftmaxOperation.apply(Tensor(pert_minus), axis=axis).data

        L_plus = np.sum(lp_plus * upstream)
        L_minus = np.sum(lp_minus * upstream)
        fd[idx] = (L_plus - L_minus) / (2 * eps)

    assert np.allclose(fd, logits._grad, atol=1e-6)
    assert np.allclose(logits._grad.sum(axis=axis, keepdims=True), 0.0, atol=1e-7)
