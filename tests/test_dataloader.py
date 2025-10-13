import numpy as np
from autograd.tensor import Tensor
from autograd.utils.data import TensorDataset, DataLoader

def test_epoch_coverage_no_drop_last():
    X = Tensor(np.arange(11), requires_grad=False)
    Y = Tensor(np.arange(11)*2, requires_grad=False)
    ds = TensorDataset(X, Y)
    seen = set()
    for xb, _ in DataLoader(ds, batch_size=3, shuffle=True, drop_last=False, seed=0):
        for i in xb.data.reshape(-1):
            seen.add(int(i))
    assert seen == set(range(11))

def test_batch_sizes_and_last_batch():
    X = Tensor(np.arange(10), requires_grad=False)
    ds = TensorDataset(X)
    batches = [xb for xb, in DataLoader(ds, batch_size=4, shuffle=False, drop_last=False)]
    sizes = [len(b.data.reshape(-1)) for b in batches]
    assert sizes == [4, 4, 2]

def test_deterministic_with_seed():
    X = Tensor(np.arange(8), requires_grad=False)
    ds = TensorDataset(X)
    order1 = [int(i) for xb, in DataLoader(ds, batch_size=2, shuffle=True, seed=123)
              for i in xb.data.reshape(-1)]
    order2 = [int(i) for xb, in DataLoader(ds, batch_size=2, shuffle=True, seed=123)
              for i in xb.data.reshape(-1)]
    assert order1 == order2