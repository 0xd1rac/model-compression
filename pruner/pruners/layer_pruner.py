import torch 
import torch.nn as nn
class LayerPruner:
    def __init__(self, criterion):
        """
        Initialize the LayerPruner with a specific pruning criterion.

        Args:
            criterion: The pruning criterion to apply at the layer level.
        """
        self.criterion = criterion

    def prune(self, layer: nn.Module) -> torch.Tensor:
        """
        Apply the pruning criterion to a specific layer.

        Args:
            layer (nn.Module): The layer to prune.

        Returns:
            torch.Tensor: A binary mask indicating which weights to prune.
        """
        if hasattr(layer, 'weight') and layer.weight is not None:
            # Apply the pruning criterion
            mask = self.criterion.apply(layer.weight)
            layer.weight.data *= mask  # Apply the mask to prune the weights
            return mask
        else:
            raise ValueError("The layer does not have a weight parameter.")
