from abc import ABC, abstractmethod
import torch
import torch.nn as nn
class BasePruningMethod(ABC):
    """
    Abstract base class for creation of new pruning techniques.
    """

    def __call__(self, module, inputs):
        pass

    @abstractmethod
    def compute_pruning_mask(self, input_tensor: torch.Tensor, default_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute and return a mask for the input tensor, input_tensor.

        This method should be implemented by subclasses to define how the mask for the 
        given `input_tensor` is computed. The method typically involves determining 
        which elements of the `input_tensor` should be pruned (set to zero) and 
        updating the `default_mask` accordingly.

        Args:
            input_tensor (torch.Tensor): The original tensor to be pruned. This could 
                be, for example, the weights of a layer in a neural network.

            default_mask (torch.Tensor): The current mask for the input_tensor, which 
                will be modified based on the pruning strategy. The default mask is 
                often initialized to all ones (i.e., no pruning), and the method updates 
                this mask based on the pruning criteria.

        Returns:
            torch.Tensor: A new mask that is applied to the `input_tensor` to prune 
            certain elements. The output mask will have the same shape as 
            `input_tensor`, with values of 1 indicating elements to keep and values 
            of 0 indicating elements to prune (set to zero).

        Notes:
            - This method should be overridden in any subclass of BasePruningMethod 
            to specify the pruning criteria.
            - The pruning could be based on different metrics such as magnitude 
            (pruning the smallest weights) or a custom strategy.
        """

    def apply_mask(self, module:nn.Module) -> torch.Tensor:
        """
        Multiplies the original tensor by the pruning mask to apply pruning.

        This function fetches the mask and the original tensor (weights) from the module
        and returns a pruned version of the tensor, where certain elements are masked (set to zero).

        Args:
            module (nn.Module): The layer or module that contains the tensor to be pruned.

        Returns:
            pruned_tensor (torch.Tensor): The pruned version of the tensor with some values masked.
        """
        
        # Ensure the tensor name is set before applying the mask
        assert self._tensor_name is not None, f"Tensor name must be set before applying pruning on module {module}"
        
        # Construct attribute names for mask and original tensor based on the tensor name
        mask_attr_name = f"{self._tensor_name}_mask"
        orig_tensor_attr_name = f"{self._tensor_name}_orig"
        
        # Retrieve the mask and original tensor from the module
        mask = getattr(module, mask_attr_name)
        orig_tensor = getattr(module, orig_tensor_attr_name)
        
        # Ensure that mask and original tensor are compatible for element-wise multiplication
        assert mask.shape == orig_tensor.shape, "Mask and original tensor must have the same shape"
        
        # Apply the mask by element-wise multiplication (zeroing out pruned elements)
        pruned_tensor = mask.to(dtype=orig_tensor.dtype) * orig_tensor
        
        return pruned_tensor
    
    @classmethod
    def apply(cls, module):
         

