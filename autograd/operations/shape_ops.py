from typing import TYPE_CHECKING, Tuple
from autograd.operation import Operation
from autograd.context import Context
import numpy as np

if TYPE_CHECKING:
    from autograd.tensor import Tensor


class ReshapeOperation(Operation):
    @classmethod
    def forward(cls, ctx: Context, tensor: "Tensor", reshape_into: Tuple) -> np.ndarray:
        ctx.save_for_backward(tensor)
        ctx.save_for_backward_values(tensor.data.shape)

        x = tensor.data

        return x.reshape(reshape_into)

    @classmethod
    def backward(cls, ctx: Context, grad_output):
        original_shape = ctx.saved_values[0]
        grad_input = grad_output.reshape(original_shape)

        return [grad_input]
