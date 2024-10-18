import torch 
import numpy as np
class PercentileCriterion:
    def __init__(self, bottom_pct:float,top_pct:float):
        """
        Initialize the percentile pruning criterion.
        
        Args:
            bottom_pct (float): The bottom percentile below which weights will be pruned (e.g., 0.1 for bottom 10%).
            top_pct (float): The top percentile above which weights will be pruned (e.g., 0.9 for top 10%).
        """
        assert 0.0 <= bottom_pct <= 1.0, "bottom_pct must be between 0 and 1"
        assert 0.0 <= top_pct <= 1.0, "top_pct must be between 0 and 1"
        assert bottom_pct < top_pct, "bottom_pct must be less than top_pct"

        self.bottom_pct = bottom_pct
        self.top_pct = top_pct

    def apply(self, weights:torch.Tensor) -> torch.Tensor:
        """
        Apply the percentile-based pruning criterion to the weights.
        
        Args:
            weights (torch.Tensor): The weights of the layer.
        
        Returns:
            torch.Tensor: A binary mask indicating which weights should be kept (1 for keep, 0 for prune).
        """

        # Initialize the mask to all True (keep all weights initially)
        mask = torch.ones_like(weights, dtype=torch.bool)

        # Flatten the weights and calculate the absolute values
        flat_weights = weights.flatten()

        # Compute the thresholds for the bottom and top percentiles
        bottom_threshold = np.percentile(flat_weights.cpu().numpy(), self.bottom_pct * 100)
        top_threshold = np.percentile(flat_weights.cpu().numpy(), self.top_pct * 100)

        # Create a mask to keep weights that are within the percentile range
        mask = (flat_weights >= bottom_threshold) & (flat_weights <= top_threshold)

        # Reshape the mask back to the original weight shape
        mask = mask.view_as(weights)

        return mask.float()  # Convert to float (1.0 for retain, 0.0 for prune)



