from autograd.optimizer import SGD

def test_step_lr_updates_every_k_steps():
    from autograd.utils.lr import StepLR
    opt = SGD([], lr=0.1)
    sch = StepLR(opt, step_size=3, gamma=0.5)
    lrs = []
    for _ in range(1, 10):
        sch.step()
        lrs.append(opt.lr)
    assert lrs == [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025, 0.025]

def test_cosine_anneal_reaches_near_zero():
    from autograd.utils.lr import CosineAnnealingLR
    opt = SGD([], lr=0.1)
    T = 10
    sch = CosineAnnealingLR(opt, T_max=T)
    vals = []
    for _ in range(T):
        sch.step()
        vals.append(opt.lr)
    assert vals[0] > vals[-1] and vals[-1] < 1e-3