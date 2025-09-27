import numpy as np
from autograd.context import Context
from autograd.operation import Operation

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autograd.tensor import Tensor

class AvgPool2dOperation(Operation):
    @classmethod
    def forward(cls, ctx: Context, input: 'Tensor', pool_size=2) -> np.ndarray:
        ctx.save_for_backward(input)
        ctx.save_for_backward_values(pool_size)

        input_h, input_w = input.data.shape
        pool_h = pool_size
        pool_w = pool_size
        output_h = input_h // pool_h
        output_w = input_w // pool_w

        output = np.zeros((output_h, output_w))

        for y in range(output_h):
            for x in range(output_w):
                pool_window = input.data[y*pool_h:(y+1)*pool_h, x*pool_w:(x+1)*pool_w]
                output[y, x] = np.mean(pool_window)

        return output

    @classmethod
    def backward(cls, ctx: Context, grad_output):
        input_tensor = ctx.saved_tensors[0]
        pool_size = ctx.saved_values[0]
        pool_h = pool_size
        pool_w = pool_size

        window_size = pool_h * pool_w
        grad_input = np.zeros_like(input_tensor.data)

        output_h, output_w = grad_output.shape

        for y in range(output_h):
            for x in range(output_w):
                gradient_per_element = grad_output[y, x] / window_size
                for i in range(pool_h):
                    for j in range(pool_w):
                        global_row = y * pool_h + i
                        global_col = x * pool_w + j
                        grad_input[global_row, global_col] = gradient_per_element


        
        return [grad_input]