import numpy as np
from autograd.tensor import Tensor
from autograd.operations.shape_ops import ReshapeOperation


def test_reshape_forward():
    """Test that reshape changes tensor shape correctly."""
    # Create a 2x3 tensor
    x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)

    # Reshape to 3x2
    y = x.reshape((3, 2))

    expected = np.array([[1, 2], [3, 4], [5, 6]])

    assert y.data.shape == (3, 2)
    assert np.array_equal(y.data, expected)


def test_reshape_same_shape():
    x = Tensor(np.arange(6).reshape(2, 3), requires_grad=True)
    y = ReshapeOperation.apply(x, (2, 3))  # same shape

    y.backward(np.ones_like(y.data))

    assert x._grad.shape == (2, 3)
    assert np.allclose(x._grad, np.ones((2, 3)))


def test_reshape_backward_gradient_flow():
    """Test that gradients flow correctly through reshape."""
    # Create input tensor (2x3)
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)

    # Reshape to (3, 2)
    y = ReshapeOperation.apply(x, (3, 2))

    # Create upstream gradient (same shape as y)
    # Simulating: some loss function computed d_loss/d_y
    upstream_grad = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    # Backward pass with upstream gradient
    y.backward(upstream_grad)

    # Gradient should have same shape as original input (2x3)
    assert x._grad.shape == (2, 3)

    # Gradient should be reshaped version of upstream grad
    expected_grad = upstream_grad.reshape(2, 3)
    assert np.allclose(x._grad, expected_grad)


def test_reshape_preserves_data():
    """Test that reshape preserves all data elements (no data loss)."""
    # Create a 4x3 tensor with unique values
    data = np.arange(12).reshape(4, 3).astype(np.float32)
    x = Tensor(data, requires_grad=True)

    # Reshape to different shapes
    y1 = ReshapeOperation.apply(x, (2, 6))
    y2 = ReshapeOperation.apply(x, (12, 1))
    y3 = ReshapeOperation.apply(x, (3, 4))

    # All should contain same elements in same order (row-major)
    assert np.array_equal(y1.data.flatten(), data.flatten())
    assert np.array_equal(y2.data.flatten(), data.flatten())
    assert np.array_equal(y3.data.flatten(), data.flatten())
