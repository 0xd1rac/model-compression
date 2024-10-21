import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from .pruning_criterion import PruningCriterion
from .weights_manager import WeightsManager

class MagnitudeCriterion(PruningCriterion):
    def __init__(self, min_threshold: float = None, max_threshold: float = None):
        """
        Magnitude-based pruning criterion that prunes weights based on their absolute values.
        It zeroes out weights smaller than the `min_threshold` or larger than the `max_threshold`.
        
        Args:  
            min_threshold (float, optional): Threshold below which weights will be pruned.
            max_threshold (float, optional): Threshold above which weights will be pruned.
        
        Raises:
            AssertionError: If both thresholds are None, or if `min_threshold` is not less than `max_threshold`.
        """
        assert min_threshold is not None or max_threshold is not None, \
            "Either min_threshold or max_threshold must be set."
        if min_threshold is not None and max_threshold is not None:
            assert min_threshold < max_threshold, "min_threshold must be smaller than max_threshold."

        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def get_mask_for_tensor(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Apply the magnitude pruning criterion to a single weight tensor.
        Weights below `min_threshold` or above `max_threshold` will be pruned.

        Args:
            weights (torch.Tensor): The weight tensor to which pruning is applied.

        Returns:
            torch.Tensor: A boolean mask of the same shape as `weights`, where `False` indicates
                          weights to prune and `True` indicates weights to keep.
        """
        flat_weights = weights.flatten()
        abs_weights = torch.abs(flat_weights)

        # Initialize mask as all True
        mask = torch.ones_like(flat_weights, dtype=torch.bool)  # Initialize mask with all `True` values (keep all weights)

        # Apply pruning based on the minimum threshold (prune weights smaller than `min_threshold`)
        if self.min_threshold is not None:
            mask &= (abs_weights > self.min_threshold)
        
        # Apply pruning based on the maximum threshold (prune weights larger than `max_threshold`)
        if self.max_threshold is not None:
            mask &= (abs_weights < self.max_threshold)
       
        # Return the boolean mask where False indicates weights to prune and True indicates weights to keep
        return mask.view_as(weights)

    def get_mask_for_model(self, model: nn.Module) -> Dict[nn.Module, torch.Tensor]:
        """
        Apply the magnitude pruning criterion to all layers of the model.
        This method collects weights from the entire model, applies the pruning criterion,
        and returns masks for each layer.

        Args:
            model (torch.nn.Module): The neural network model to prune.

        Returns:
            Dict[nn.Module, torch.Tensor]: A dictionary where the keys are model layers (modules),
                                           and the values are the corresponding boolean masks for the layer's weights.
        """
        # Collect all model weights into a single concatenated tensor and track the parameter mappings
        all_weights, param_mapping = WeightsManager.collect_model_weights(model)

        # Apply the magnitude-based pruning criterion to the collected weights
        tensor_mask = self.get_mask_for_tensor(all_weights)

        # Build individual masks for each module from the global tensor mask
        module_to_mask = WeightsManager.build_masks(tensor_mask, param_mapping)
        
        return module_to_mask