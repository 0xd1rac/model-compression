import torch
import numpy as np
import torch.nn as nn
from typing import Dict
from .weights_manager import WeightsManager

class PercentileCriterion:
    def __init__(self, bottom_pct: float = None, top_pct: float = None):
        """
        Initialize the percentile pruning criterion for global pruning.
        
        Args:
            bottom_pct (float, optional): The bottom percentile below which weights will be pruned 
                                          (e.g., 0.1 for bottom 10%).
            top_pct (float, optional): The top percentile above which weights will be pruned 
                                       (e.g., 0.9 for top 10%).
        
        Raises:
            AssertionError: If either percentile is not between 0 and 1, or if bottom_pct >= top_pct.
        """
        if bottom_pct is not None:
            assert 0.0 <= bottom_pct <= 1.0, "bottom_pct must be between 0 and 1"
        
        if top_pct is not None:
            assert 0.0 <= top_pct <= 1.0, "top_pct must be between 0 and 1"
        
        if bottom_pct is not None and top_pct is not None:
            assert bottom_pct < top_pct, "bottom_pct must be less than top_pct"

        self.bottom_pct = bottom_pct
        self.top_pct = top_pct

    def get_mask_for_tensor(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Apply the percentile-based pruning criterion to a weight tensor.
        Prune weights in the bottom or top percentile based on absolute values.

        Args:
            weights (torch.Tensor): The weight tensor to prune.

        Returns:
            torch.Tensor: A boolean mask where `False` represents pruned weights 
                          and `True` represents kept weights.
        """
        # Flatten the tensor and get absolute values of the weights
        flat_weights = weights.flatten()
        abs_weights = torch.abs(flat_weights)

        # Initialize mask as all True
        total_weights = weights.numel()
        mask = torch.ones(total_weights, dtype=torch.bool)

        # Calculate and apply bottom percentile pruning
        if self.bottom_pct is not None:
            bottom_threshold = np.percentile(abs_weights.cpu().numpy(), self.bottom_pct * 100)
            mask &= (abs_weights > bottom_threshold)

        # Calculate and apply top percentile pruning
        if self.top_pct is not None:
            top_threshold = np.percentile(abs_weights.cpu().numpy(), self.top_pct * 100)
            mask &= (abs_weights < top_threshold)

        # Reshape the mask to the same shape as the original weights
        return mask.view_as(weights)

    def get_mask_for_model(self, model: nn.Module) -> Dict[nn.Module, torch.Tensor]:
        """
        Apply the percentile-based pruning criterion to all weights in the model.

        Args:
            model (torch.nn.Module): The neural network model.

        Returns:
            Dict[nn.Module, torch.Tensor]: A dictionary where the keys are layers (modules) 
                                           and the values are the boolean masks for each layer's weights.
        """
        # Collect all model weights into a single concatenated tensor and get parameter mappings
        all_weights, param_mapping = WeightsManager.collect_model_weights(model)

        # Apply the percentile-based pruning to the global tensor of weights
        tensor_mask = self.get_mask_for_tensor(all_weights)

        # Build individual masks for each module from the global tensor mask
        module_to_mask = WeightsManager.build_masks(tensor_mask, param_mapping)
        
        return module_to_mask
