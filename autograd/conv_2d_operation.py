from typing import TYPE_CHECKING, List
from autograd.operation import Operation
from autograd.context import Context
import numpy as np

if TYPE_CHECKING:
    from autograd.tensor import Tensor

class Conv2DOperation(Operation):
    @classmethod
    def forward(cls, ctx: Context, input: 'Tensor', kernel: 'Tensor') -> np.ndarray:
        ctx.save_for_backward(input, kernel)
        input_h, input_w = input.data.shape
        kernel_h, kernel_w = kernel.data.shape

        output_h = input_h - kernel_h + 1
        output_w = input_w - kernel_w + 1

        output = np.zeros((output_h, output_w))

        for y in range(output_h):
            for x in range(output_w):
                input_region = input.data[y:y+kernel_h, x:x+kernel_w]
                output[y, x] = np.sum(input_region * kernel.data)
        return output

    @classmethod
    def backward(cls, ctx, grad_output):
        input, kernel = ctx.saved_tensors

        output_h, output_w = grad_output.shape
        kernel_h, kernel_w = kernel.data.shape

        grad_input = np.zeros_like(input.data)
        grad_kernel = np.zeros_like(kernel.data)

        for y in range(output_h):
            for x in range(output_w):
                # input window based on kernel moving window
                input_region = input.data[y:y+kernel_h, x:x+kernel_w]
                # [[1, 1], [1, 1]] (default gradient) * moving kernel window 2x2
                grad_kernel += grad_output[y, x] * input_region

                grad_input[y:y+kernel_h, x:x+kernel_w] += grad_output[y, x] * kernel.data

        return [grad_input, grad_kernel]

class FlattenOperation(Operation):
    """Flatten operation that preserves gradients."""
    
    @classmethod
    def forward(cls, ctx: Context, input_tensor: 'Tensor') -> np.ndarray:
        ctx.save_for_backward(input_tensor)
        original_shape = input_tensor.data.shape
        ctx.save_for_backward_values(original_shape)
        
        return input_tensor.data.flatten().reshape(-1, 1)
    
    @classmethod  
    def backward(cls, ctx: Context, grad_output: np.ndarray) -> List[np.ndarray]:
        original_shape, = ctx.saved_values
        
        grad_input = grad_output.flatten().reshape(original_shape)
        return [grad_input]