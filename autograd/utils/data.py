from autograd.tensor import Tensor

import numpy as np

class TensorDataset:
    def __init__(self, *tensors) -> None:
        n = len(tensors[0]._data)
        for t in tensors[1:]:
            assert len(t._data) == n
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors[0]._data)

    def __getitem__(self, idx):
        return tuple(Tensor(t._data[idx], requires_grad=t._requires_grad) for t in self.tensors)

class DataLoader:
    def __init__(self, ds, batch_size=32, shuffle=True, drop_last=False, seed=None) -> None:
        self.ds = ds
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

    def __iter__(self):
        idx = np.arange(len(self.ds))
        if self.shuffle:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(idx)

        for i in range(0, len(idx), self.batch_size):
            batch_idx = idx[i:i+self.batch_size]
            if self.drop_last == True and len(batch_idx) < self.batch_size:
                break

            samples = [self.ds[j] for j in batch_idx]

            fields = list(zip(*samples)) if samples else []

            batches = []
            for field in fields:
                data_list = [t._data for t in field]
                batch_arr = np.stack(data_list, axis=0)

                req = any(t._requires_grad for t in field)
                batches.append(Tensor(batch_arr, requires_grad=req))
            
            yield tuple(batches)
