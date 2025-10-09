from math import log
import numpy as np
import pytest
from autograd.operations.nll_loss import NLLLoss
from autograd.tensor import Tensor
from autograd.operations.log_softmax import LogSoftmaxOperation

# CrossEntropyWithLogits(logits, target, axis) = NLLLoss(LogSoftmax(logits, axis), target, axis)
def test_nll_forward_one_hot_axis0():
    logits = Tensor([[1.0, -2.0],
                     [0.0,  3.0],
                     [2.0,  1.0]])
    target = Tensor([[1.0, 0.0],
                     [0.0, 1.0],
                     [0.0, 0.0]])
    log_probs = LogSoftmaxOperation.apply(logits, axis=0).data
    loss = NLLLoss.apply(log_probs, target, axis=0)

    expected = -np.sum(log_probs.data * target.data, axis=0)
    assert np.allclose(loss.data, expected, atol=1e-7)

def test_nll_forward_shift_axis0():
    logits = Tensor([[1.0, -2.0],
                     [0.0,  3.0],
                     [2.0,  1.0]])
    target = Tensor([[1.0, 0.0],
                     [0.0, 1.0],
                     [0.0, 0.0]]) 
    log_probs = LogSoftmaxOperation.apply(logits, axis = 0).data
    loss = NLLLoss.apply(log_probs, target, axis= 0)

    logits_shifted = Tensor(logits.data + 10.0)
    log_probs_shifted = LogSoftmaxOperation.apply(logits_shifted, axis = 0).data
    loss_shifted = NLLLoss.apply(log_probs_shifted, target, axis = 0)

    assert np.allclose(loss.data, loss_shifted.data, atol=1e-7)

def test_nll_forward_one_hot_axis1():
    logits = Tensor([[ 0.5, -1.0, 2.0],
                     [-0.5,  1.5, 0.0]])
    target = Tensor([[0.0, 0.0, 1.0],
                     [0.0, 1.0, 0.0]])

    log_probs = LogSoftmaxOperation.apply(logits, axis=1)
    loss = NLLLoss.apply(log_probs, target, axis=1)

    expected = -np.sum(log_probs.data * target.data, axis=1, keepdims=True)
    assert np.allclose(loss.data, expected, atol=1e-7)

def test_nll_backward_one_hot_axis0():
    logits = Tensor([[1.0, -2.0],
                     [0.0,  3.0],
                     [2.0,  1.0]], requires_grad=True)
    y = Tensor([[1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0]]) 
    
    log_probs = LogSoftmaxOperation.apply(logits, axis=0)
    loss = NLLLoss.apply(log_probs, y, axis=0)

    upstream = np.ones_like(loss.data)
    loss.backward(upstream)

    # expected: grad_logits = softmax(logits) - y
    probs = np.exp(log_probs.data)
    expected = probs - y.data

    assert np.allclose(logits._grad, expected, atol=1e-7)

@pytest.mark.parametrize("axis", [0, 1])
def test_tensor_ce_wrapper(axis):
    logits = Tensor([[ 0.5, -1.0, 2.0],
                     [-0.5,  1.5, 0.0]])
    target = Tensor([[0.0, 0.0, 1.0],
                     [0.0, 1.0, 0.0]])
    ce = logits.cross_entropy_with_logits(target, axis)
    expected = NLLLoss.apply(LogSoftmaxOperation.apply(logits, axis=axis), target, axis=axis)

    assert np.allclose(ce.data, expected.data, atol=1e-7)

@pytest.mark.parametrize("axis", [0, 1])
def test_ce_with_logits_backward(axis):
    logits = Tensor([[1.0, -2.0],
                     [0.0,  3.0],
                     [2.0,  1.0]], requires_grad=True) if axis == 0 else \
             Tensor([[ 0.5, -1.0, 2.0],
                     [-0.5,  1.5, 0.0]], requires_grad=True)

    y = Tensor([[1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0]]) if axis == 0 else \
        Tensor([[0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0]])

    ce = logits.cross_entropy_with_logits(y, axis=axis)
    upstream = np.ones_like(ce.data)
    ce.backward(upstream)

    # expected grad: softmax(logits) - y, along the same axis
    log_probs = LogSoftmaxOperation.apply(logits, axis=axis).data
    probs = np.exp(log_probs)
    expected = probs - y.data

    assert np.allclose(logits._grad, expected, atol=1e-7)


