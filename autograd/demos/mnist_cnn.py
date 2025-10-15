from autograd.adam import Adam
from autograd.models.simple_cnn import SimpleCNN
from autograd.operations.log_softmax import LogSoftmaxOperation
from autograd.operations.nll_loss import NLLLoss
from autograd.utils.data import TensorDataset, DataLoader
from autograd.tensor import Tensor
from autograd.utils.lr import StepLR
from autograd.optimizer import SGD

from mnist import MNIST
import numpy as np

def one_hot(y, num_classes=10):
    oh = np.zeros((num_classes, y.size), dtype=np.float32)
    oh[y, np.arange(y.size)] = 1.0
    return oh

def load_mnist_python_mnist(root="data/mnist"):
    mndata = MNIST(root)
    mndata.gz = True 
    Xtr, ytr = mndata.load_training()
    Xte, yte = mndata.load_testing()

    Xtr = (np.asarray(Xtr, dtype=np.float32) / 255.0).reshape(-1, 28*28).T
    Xte = (np.asarray(Xte, dtype=np.float32) / 255.0).reshape(-1, 28*28).T
    ytr = np.asarray(ytr, dtype=np.int64)
    yte = np.asarray(yte, dtype=np.int64)

    Ytr = one_hot(ytr, 10) 
    Yte = one_hot(yte, 10)
    return Xtr, Ytr, Xte, Yte

def val_accuracy(model, val_loader):
    correct = total = 0
    for xb, yb in val_loader:
        for i in range(xb.data.shape[0]):
            logits = model(Tensor(xb.data[i], False))
            pred = int(np.argmax(logits.data))
            true = int(np.argmax(yb.data[i]))
            correct += int(pred == true)
            total += 1
    return correct / max(total, 1)

Xtr, Ytr, Xte, Yte = load_mnist_python_mnist()

Xtr_imgs = Xtr.T.reshape(-1, 28, 28)
Ytr_onehot = Ytr.T

N_train = 10000
Xtr_imgs_sub = Xtr_imgs[:N_train]
Ytr_onehot_sub = Ytr_onehot[:N_train]
train_ds = TensorDataset(Tensor(Xtr_imgs_sub, False), Tensor(Ytr_onehot_sub, False))
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, seed=0)

Xte_imgs = Xte.T.reshape(-1, 28, 28)
Yte_onehot = Yte.T
N_val = 4000
Xte_imgs_sub = Xte_imgs[:N_val]
Yte_onehot_sub = Yte_onehot[:N_val]
val_ds = TensorDataset(Tensor(Xte_imgs_sub, False), Tensor(Yte_onehot_sub, False))
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

model = SimpleCNN()
opt = SGD(model.parameters(), momentum=0.9, lr=0.05, weight_decay=1e-4)
sch = StepLR(opt, step_size=3, gamma=0.5)

for ep in range(10):
    total, count = 0.0, 0
    for xb, yb in train_loader:
        opt.zero_grad()
        batch_loss = 0.0
        B = xb.data.shape[0]
        for i in range(B):
            img = Tensor(xb.data[i], False)
            tgt = Tensor(yb.data[i].reshape(10,1), False)
            logits = model(img)
            logp = LogSoftmaxOperation.apply(logits, axis=0)
            loss = NLLLoss.apply(logp, tgt, axis=0)
            batch_loss += float(loss.data.item())
            (loss * (1.0 / B)).backward()
        opt.step()
        total += batch_loss / B
        count += 1
    sch.step()
    acc = val_accuracy(model, val_loader)
    print(f"epoch {ep+1}: avg_loss={total/max(count,1):.4f}, lr={opt.lr:.5f}, val_acc={acc:.3f}")

