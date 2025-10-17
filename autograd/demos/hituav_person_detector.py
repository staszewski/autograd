from pathlib import Path
from autograd.datasets.hit_uav_patches import build_patches
from autograd.tensor import Tensor
from autograd.utils.data import TensorDataset, DataLoader
import numpy as np
from autograd.models.simple_cnn import SimpleCNN
from autograd.adam import Adam
from autograd.operations.log_softmax import LogSoftmaxOperation
from autograd.operations.nll_loss import NLLLoss
from autograd.utils.lr import StepLR

root = str((Path(__file__).resolve().parents[2] / "data" / "hit-uav"))

Xtr, Ytr = build_patches(root, "train",
                         n_neg_per_pos=2, max_pos_per_image=4,
                         max_images=400, max_total=15000, rng_seed=0)
Xva, Yva = build_patches(root, "val",   n_neg_per_pos=1, rng_seed=1)

train_loader = DataLoader(TensorDataset(Tensor(Xtr, False), Tensor(Ytr, False)),
                          batch_size=64, shuffle=True, seed=0)
val_loader   = DataLoader(TensorDataset(Tensor(Xva, False), Tensor(Yva, False)),
                          batch_size=256, shuffle=False)

model = SimpleCNN(K=4, num_classes=2)
opt = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
sch = StepLR(opt, step_size=3, gamma=0.5)

epochs = 10
for ep in range(epochs):
    total = count = 0
    for xb, yb in train_loader:
        B = xb.data.shape[0]
        opt.zero_grad()
        batch_loss = 0.0
        for i in range(B):
            img = Tensor(xb.data[i], False)
            tgt = Tensor(yb.data[i].reshape(2,1), False)
            logits = model(img)
            logp = LogSoftmaxOperation.apply(logits, axis=0)
            loss = NLLLoss.apply(logp, tgt, axis=0)
            batch_loss += float(loss.data.item())
            (loss * (1.0 / B)).backward()
        opt.step()
        total += batch_loss / B; count += 1
    sch.step()

    correct = 0
    tot = 0
    for xb, yb in val_loader:
        for i in range(xb.data.shape[0]):
            pred = int(np.argmax(model(Tensor(xb.data[i], False)).data, axis=0).item())
            true = int(np.argmax(yb.data[i]))
            correct += int(pred == true); tot += 1
    print(f"epoch {ep+1}: loss={total/max(count,1):.4f}, lr={opt.lr:.5f}, val_acc={correct/max(tot,1):.3f}")