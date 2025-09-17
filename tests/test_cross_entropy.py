from autograd import Tensor, CrossEntropyLoss
import numpy as np

def test_cross_entropy_forward():
    predictions = Tensor([0.9, 0.1], requires_grad=False)
    targets = Tensor([1.0, 0.0], requires_grad=False)
    loss = CrossEntropyLoss.apply(predictions, targets)

    assert loss.data is not None
    # -[1*log(0.9) + 0*log(0.1)] = -log(0.9) ≈ 0.10536
    expected_loss = -np.log(0.9)
    assert np.allclose(loss.data, expected_loss, atol=1e-6)

def test_cross_entropy_backward():
    predictions = Tensor([0.99, 0.01], requires_grad=True)  
    targets = Tensor([1.0, 0.0], requires_grad=False)
    loss = CrossEntropyLoss.apply(predictions, targets)
    loss.backward()

    assert np.allclose(loss.data, 0.01005, atol=1e-4)

    expected_grad = [-0.505, 0.505]
    assert np.allclose(predictions._grad, expected_grad, atol=1e-3)

    assert np.array_equal(targets._grad, np.zeros_like(targets.data))