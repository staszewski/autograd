import numpy as np
from autograd.tensor import Tensor
from autograd.optimizer import SGD
from autograd.adam import Adam

def loss_fn(w, w_star=2.0):
    # L = 0.5 * (w - w*)^2
    diff = w - Tensor([[w_star]], requires_grad=False)
    return 0.5 * (diff * diff)

def run_opt(w, optimizer, steps=100, grad_clip=None):
    hist = []
    for _ in range(steps):
        l = loss_fn(w)
        optimizer.zero_grad()
        l.backward()
        if isinstance(optimizer, SGD):
            optimizer.step(grad_clip=grad_clip)
        else:
            optimizer.step()
        hist.append(float(loss_fn(w).data.item()))
    return w, hist

def main():
    w_sgd = Tensor(np.array([[5.0]], dtype=np.float32), requires_grad=True)
    w_mom = Tensor(np.array([[5.0]], dtype=np.float32), requires_grad=True)
    w_adam = Tensor(np.array([[5.0]], dtype=np.float32), requires_grad=True)

    sgd = SGD([w_sgd], lr=0.1)
    mom = SGD([w_mom], lr=0.1, momentum=0.9)
    adam = Adam([w_adam], lr=0.05) 

    _, h_sgd = run_opt(w_sgd, sgd, steps=80)
    _, h_mom = run_opt(w_mom, mom, steps=80)
    _, h_adam = run_opt(w_adam, adam, steps=80)

    print("final |w - w*|:",
          float(np.abs(w_sgd.data - 2.0).item()),
          float(np.abs(w_mom.data - 2.0).item()),
          float(np.abs(w_adam.data - 2.0).item()))
    print("last losses:",
          h_sgd[-1], h_mom[-1], h_adam[-1])

    w_wd = Tensor(np.array([[5.0]], dtype=np.float32), requires_grad=True)
    w_clip = Tensor(np.array([[5.0]], dtype=np.float32), requires_grad=True)

    sgd_wd = SGD([w_wd], lr=0.1, momentum=0.9, weight_decay=0.01)
    _, h_wd = run_opt(w_wd, sgd_wd, steps=80)
    print("with weight decay, last loss:", h_wd[-1])

    sgd_clip = SGD([w_clip], lr=0.5)
    _, h_clip = run_opt(w_clip, sgd_clip, steps=10, grad_clip=1.0)
    print("with grad clip, last loss:", h_clip[-1])

if __name__ == "__main__":
    main()