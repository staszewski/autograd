import pytest
import numpy as np
from autograd.tensor import Tensor
from autograd.operations.softmax import SoftmaxOperation

def test_softmax_forward_sums_to_one_and_shift_invariance():
    logits = Tensor([[1.0, -2.0],
                     [0.0,  3.0],
                     [2.0,  1.0]])
    probs = SoftmaxOperation.apply(logits)

    col_sums = probs.data.sum(axis=0, keepdims=True)

    assert np.allclose(col_sums, np.ones_like(col_sums))

    shift = Tensor([[10.0], [10.0], [10.0]])
    logits_shifted = Tensor(logits.data + shift.data)
    probs_shifted = SoftmaxOperation.apply(logits_shifted)

    assert np.allclose(probs.data, probs_shifted.data, atol=1e-7)

def test_softmax_forward_generalize_axis():
    logits = Tensor([[1.0, -2.0],
                     [0.0,  3.0],
                     [2.0,  1.0]])
    axis = 1
    probs = SoftmaxOperation.apply(logits, axis = axis)

    col_sums = probs.data.sum(axis=axis, keepdims=True)

    assert np.allclose(col_sums, np.ones_like(col_sums))

    shift = Tensor([[10.0], [10.0], [10.0]])
    logits_shifted = Tensor(logits.data + shift.data)
    probs_shifted = SoftmaxOperation.apply(logits_shifted, axis = axis)

    assert np.allclose(probs.data, probs_shifted.data, atol=1e-7)

def test_softmax_backward_matches_vector_jacobian_form():
    x = Tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    y = SoftmaxOperation.apply(x)

    upstream = np.array([[0.2], [-0.1], [0.7]])
    y.backward(upstream)

    probs = y.data
    s = (upstream * probs).sum(axis=0, keepdims=True)
    expected = probs * (upstream - s)

    assert np.allclose(x._grad, expected, atol=1e-7)
    assert np.allclose(x._grad.sum(axis=0, keepdims=True), 0.0, atol=1e-7)

@pytest.mark.parametrize("axis", [0, 1])
def test_finite_difference_grad_check(axis):
    np.random.seed(42)
    dim = 3
    if axis == 0:
        class_batch = np.random.randn(dim, 1)
        logits_data = class_batch 
    else:
        batch_class = np.random.randn(1, dim)
        logits_data = batch_class 
    logits = Tensor(logits_data, requires_grad=True)

    probs = SoftmaxOperation.apply(logits, axis = axis)
    up = np.random.randn(*probs.data.shape)
    probs.backward(up)

    eps = 1e-6
    grad_input_fd = np.zeros_like(logits.data)
    
    for idx in np.ndindex(logits_data.shape):
        # perturb
        logits_plus = logits_data.copy()
        logits_plus[idx] += eps
        logits_minus = logits_data.copy()
        logits_minus[idx] -= eps
        # recompute probs
        probs_plus = SoftmaxOperation.apply(Tensor(logits_plus), axis = axis).data
        probs_minus = SoftmaxOperation.apply(Tensor(logits_minus), axis = axis).data

        delta_probs = (probs_plus - probs_minus) / (2 * eps)
        grad_input_fd[idx] = np.sum(up * delta_probs)

    assert np.allclose(grad_input_fd, logits._grad, atol=1e-7)
    assert np.allclose(logits._grad.sum(axis=axis, keepdims=True), 0.0)
