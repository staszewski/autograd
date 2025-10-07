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