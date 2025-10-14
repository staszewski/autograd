import numpy as np
from mnist import MNIST
from autograd.tensor import Tensor
from autograd.mlp import MLP
from autograd.operations.log_softmax import LogSoftmaxOperation
from autograd.operations.nll_loss import NLLLoss
from autograd.optimizer import SGD

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

Xtr, Ytr, Xte, Yte = load_mnist_python_mnist()
Xtr_t, Ytr_t = Tensor(Xtr, False), Tensor(Ytr, False)
Xte_t, Yte_t = Tensor(Xte, False), Tensor(Yte, False)

mlp = MLP(input_size=784, hidden_size=256, output_size=10, activation="tanh")
opt = SGD(mlp.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
lr, batch_size, epochs = 0.05, 256, 20

def accuracy(logits, Y):
    pred = np.argmax(logits.data, axis=0)
    true = np.argmax(Y.data, axis=0)
    return np.mean(pred == true)

N = Xtr.shape[1]
for ep in range(epochs):
    perm = np.random.permutation(N)
    tot_loss = 0.0
    for s in range(0, N, batch_size):
        idx = perm[s:s+batch_size]
        xb = Tensor(Xtr[:, idx], False)
        yb = Tensor(Ytr[:, idx], False)

        logits = mlp(xb)
        logp = LogSoftmaxOperation.apply(logits, axis=0)
        loss = NLLLoss.apply(logp, yb, axis=0)
        tot_loss += float(loss.data.mean())

        opt.zero_grad()
        loss.backward(np.ones_like(loss.data) / loss.data.shape[1])
        opt.step()

    num_batches = int(np.ceil(N / batch_size))
    print('Average loss', tot_loss / num_batches)
    val_acc = accuracy(mlp(Xte_t), Yte_t)
    print(f"epoch {ep+1}: train_loss≈{tot_loss:.3f}, val_acc={val_acc:.3f}")


def write_kaggle_submission(mlp, kaggle_test_csv, out_csv):
    # Load Kaggle test.csv (rows=samples, 784 columns), normalize, make columns=samples
    Xtest = np.loadtxt(kaggle_test_csv, delimiter=',', skiprows=1, dtype=np.float32)
    Xtest = (Xtest / 255.0).T  # (784, N)

    logits = mlp(Tensor(Xtest, requires_grad=False))

    preds = np.argmax(logits.data, axis=0).astype(int)
    with open(out_csv, 'w') as f:
        f.write('ImageId,Label\n')
        for i, p in enumerate(preds, 1):
            f.write(f'{i},{p}\n')

write_kaggle_submission(mlp, 'data/mnist/test.csv', 'submission.csv')