from autograd.context import Context
from autograd.operation import Operation
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autograd.tensor import Tensor


class MaxPool2dOperation(Operation):
    @classmethod
    def forward(cls, ctx: Context, input: 'Tensor', pool_size=2) -> np.ndarray:
        ctx.save_for_backward(input)
        input_h, input_w = input.data.shape
        pool_h = pool_size
        pool_w = pool_size

        output_h = input_h // pool_h
        output_w = input_w // pool_w

        max_positions = []
        output = np.zeros((output_h, output_w))

        for y in range(output_h):
            for x in range(output_w):
                pool_window = input.data[y*pool_h:(y+1)*pool_h, x*pool_w:(x+1)*pool_w]

                max_flat_index = np.argmax(pool_window)
                max_row_in_window, max_col_in_window = np.unravel_index(max_flat_index, pool_window.shape)

                global_row = y * pool_h + max_row_in_window
                global_col = x * pool_w + max_col_in_window

                max_positions.append((global_row, global_col))
                output[y, x] = pool_window[max_row_in_window, max_col_in_window] 

        ctx.save_for_backward_values(max_positions)
        return output

    @classmethod
    def backward(cls, ctx: Context, grad_output):
        input_tensor = ctx.saved_tensors[0]
        max_positions = ctx.saved_values[0]

        grad_input = np.zeros_like(input_tensor.data)

        output_h, output_w = grad_output.shape

        idx = 0

        for y in range(output_h):
            for x in range(output_w):
                global_row, global_col = max_positions[idx]
                grad_input[global_row, global_col] = grad_output[y, x]
                idx += 1
        
        return [grad_input]
