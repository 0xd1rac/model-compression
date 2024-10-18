import torch
import torch.nn as nn

class ModelPruner:
    def __init__(self, criterion):
        """
        Initialize the ModelPruner with a global pruning criterion.

        Args:
            criterion: The global pruning criterion (e.g., magnitude-based, second-order).
        """
        self.criterion = criterion

    def apply_pruning(self, model: nn.Module) -> dict:
        """
        Apply global pruning to the entire model based on the criterion.

        Args:
            model (nn.Module): The neural network model to be pruned.

        Returns:
            dict: A dictionary with layer names as keys and pruning masks as values.
        """
        masks = self.criterion.apply(model)

        # Apply the masks to prune weights globally
        for name, param in model.named_parameters():
            if name in masks:
                mask = masks[name]
                param.data *= mask  # Apply the mask to prune weights

        return masks
