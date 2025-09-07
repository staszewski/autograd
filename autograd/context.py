from typing import List, Any



class Context:
    """Context manager for saving tensors and values needed during backward pass."""
    
    def __init__(self):
        self._saved_tensors: List['Tensor'] = []
        self._saved_values: List[Any] = []
    
    def save_for_backward(self, *tensors: 'Tensor') -> None:
        """Save tensors for use in backward pass."""
        self._saved_tensors = list(tensors)
    
    def save_for_backward_values(self, *values: Any) -> None:
        """Save arbitrary values for use in backward pass."""
        self._saved_values = list(values)
    
    @property
    def saved_tensors(self) -> List['Tensor']:
        """Get saved tensors."""
        return self._saved_tensors
    
    @property
    def saved_values(self) -> List[Any]:
        """Get saved values."""
        return self._saved_values
