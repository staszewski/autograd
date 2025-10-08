import numpy as np
from autograd.context import Context
from autograd.operation import Operation

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autograd.tensor import Tensor

class LogSoftmaxOperation(Operation):
    @classmethod
    def forward(cls, ctx: Context, input: 'Tensor', axis = 0):
        ctx.save_for_backward(input)
        x = input.data
        x_max = np.max(x, axis = axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        log_sum = np.log(np.sum(exp_x, axis = axis, keepdims=True))
        log_probs = x - x_max - log_sum
        ctx.save_for_backward_values(log_probs, axis)
        return log_probs
        
    @classmethod
    def backward(cls, ctx: Context, grad_output):
        log_probs = ctx.saved_values[0]
        axis = ctx.saved_values[1]
        p = np.exp(log_probs.data)
        scalar = np.sum(grad_output, axis=axis, keepdims=True)
        grad_input = grad_output - scalar * p
        return [grad_input]
