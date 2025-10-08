import numpy as np
from autograd.context import Context
from autograd.operation import Operation

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autograd.tensor import Tensor

class SoftmaxOperation(Operation):
    @classmethod
    def forward(cls, ctx: Context, input: 'Tensor', axis = 0):
        ctx.save_for_backward(input)
        x = input.data
        x_shifted = x - np.max(x, axis = axis, keepdims = True)
        exp_x = np.exp(x_shifted)
        probs = exp_x / np.sum(exp_x, axis = axis, keepdims=True)
        ctx.save_for_backward_values(probs, axis)
        return probs

    @classmethod
    def backward(cls, ctx: Context, grad_output):
        probs = ctx.saved_values[0]
        axis = ctx.saved_values[1]
        scalar_projection = np.sum(grad_output * probs, axis=axis, keepdims=True)
        grad_input = probs * (grad_output - scalar_projection)
        return [grad_input]