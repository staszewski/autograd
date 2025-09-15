from autograd.operation import Operation
from autograd.context import Context
from autograd import Tensor
import numpy as np
from typing import List


class MSELoss(Operation):
    """Mean Squared Error loss function"""
    
    @classmethod
    def forward(cls, ctx: Context, predictions: 'Tensor', targets: 'Tensor') -> np.ndarray:
        ctx.save_for_backward(predictions, targets)
        diff = predictions.data - targets.data
        return np.mean(diff ** 2)
    
    @classmethod 
    def backward(cls, ctx: Context, grad_output: np.ndarray) -> List[np.ndarray]:
        predictions, targets = ctx.saved_tensors
        n = predictions.data.size
        
        # d/d_pred MSE = 2 * (pred - target) / n
        grad_predictions = 2 * (predictions.data - targets.data) / n * grad_output
        grad_targets = -grad_predictions
        
        return [grad_predictions, grad_targets]