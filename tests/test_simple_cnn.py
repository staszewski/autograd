import numpy as np
from autograd.tensor import Tensor
from autograd.models.simple_cnn import SimpleCNN
from autograd.operations.log_softmax import LogSoftmaxOperation
from autograd.operations.nll_loss import NLLLoss
from autograd.adam import Adam

def test_simple_cnn_forward_shape():
    np.random.seed(0)
    img = Tensor(np.random.randn(28, 28).astype(np.float32), requires_grad=False)
    model = SimpleCNN()
    logits = model(img)
    assert logits.data.shape == (10, 1)
    for p in model.parameters():
        assert p.requires_grad

def test_simple_cnn_tiny_training_decreases_loss():
    np.random.seed(1)
    img = Tensor(np.random.randn(28, 28).astype(np.float32), requires_grad=False)
    target_idx = 3
    target = np.zeros((10, 1), dtype=np.float32)
    target[target_idx, 0] = 1.0
    target_t = Tensor(target, requires_grad=False)

    model = SimpleCNN()
    opt = Adam(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(30):
        logits = model(img)
        logp = LogSoftmaxOperation.apply(logits, axis=0)
        loss = NLLLoss.apply(logp, target_t, axis=0)
        losses.append(float(loss.data.item()))
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert losses[-1] < losses[0]