import torch
import torch.nn as nn
import random
from typing import List, Dict
from .pruning_criterion import PruningCriterion
from .weights_manager import WeightsManager
from context.global_pruning_context import GlobalPruningContext

class RandomCriterion(PruningCriterion):
    def __init__(self, prune_ratio: float = 0.9):
        """
        Initialize the RandomCriterion for random pruning.

        Args:
            prune_ratio (float): The percentage of weights to prune, 
                                 should be between 0 and 1 (0.9 means 90% of weights will be pruned).
        """
        assert 0.0 < prune_ratio < 1.0, "prune_ratio must be between 0 and 1."
        self.prune_ratio = prune_ratio

    def get_mask_for_tensor(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Apply random pruning to a single weight tensor by generating a boolean mask where 
        a specified percentage of weights are randomly set to be pruned.

        Args:
            weights (torch.Tensor): The tensor of weights from a single layer or module.

        Returns:
            torch.Tensor: A boolean mask of the same shape as the input weights, 
                          where False indicates pruned weights, and True indicates kept weights.
        """
        total_weights = weights.numel()  # Get the total number of elements (weights) in the tensor
        mask = torch.ones(total_weights, dtype=torch.bool)  # Initialize a mask with all elements set to True (keep all)

        # Determine the number of weights to prune
        num_prune = int(self.prune_ratio * total_weights)

        # Randomly select indices for pruning
        prune_indices = random.sample(range(total_weights), num_prune)
        mask[prune_indices] = False  # Set the selected indices to False (prune these weights)

        # Reshape the mask to match the original weight tensor's shape
        return mask.view_as(weights)

    def get_mask_for_model(self, model: nn.Module) -> Dict[nn.Module, torch.Tensor]:
        """
        Apply random pruning to the entire model. This method treats the model's weights as 
        one large tensor, prunes them randomly based on the prune_ratio, 
        and returns a dictionary of boolean masks for each layer.

        Args:
            model (torch.nn.Module): The neural network model to apply pruning to.

        Returns:
            Dict[nn.Module, torch.Tensor]: A dictionary where each key is a module (layer), 
                                           and the value is the boolean mask for that layer's weights.
        """
        # Collect all model weights into a single concatenated tensor and get parameter mappings
        all_weights, param_mapping = WeightsManager.collect_model_weights(model)
        
        # Apply random pruning to the concatenated tensor of weights
        tensor_mask = self.get_mask_for_tensor(all_weights)
        
        # Build individual masks for each module from the global tensor mask
        module_to_mask = WeightsManager.build_masks(tensor_mask, param_mapping)
        
        return module_to_mask