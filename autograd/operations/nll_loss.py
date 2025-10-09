import numpy as np
from autograd.context import Context
from autograd.operation import Operation

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autograd.tensor import Tensor

class NLLLoss(Operation):
    @classmethod
    def forward(cls, ctx: Context, log_probs: 'Tensor', target: 'Tensor', axis = 0):
        ctx.save_for_backward(log_probs, target)
        ctx.save_for_backward_values(axis)

        l = log_probs.data
        y = target.data
        loss = -np.sum(l * y, axis=axis, keepdims=True)
        
        return loss

    @classmethod
    def backward(cls, ctx: Context, grad_output):
        log_probs, target_t = ctx.saved_tensors

        grad_log_probs = -target_t.data * grad_output
        grad_target = np.zeros_like(target_t.data)
        
        return [grad_log_probs, grad_target]
