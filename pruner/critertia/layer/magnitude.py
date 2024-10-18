
import torch 

class MagnitudeCriterion:
    def __init__(self, min_threshold:float=None, max_threshold:float=None):
        """
        Magnitude-based pruning zeroes out weights smaller than min_threshold 
        and larger that max_threshold.

        Args:  
            min_threshold (float): Threshold below which weights will be pruned.
            max_threshold (float): Threshold above which weights will be pruned.
        """
        assert min_threshold is not None or max_threshold is not None, "args, min_threshold OR max_threshold must be set."
        assert min_threshold < max_threshold, "min_threshold MUST be smaller than max_threshold."

        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
    
    def apply(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Apply the pruning criterion to the weights.

        Args:
            weights (torch.Tensor): The weights of the layer.

        Returns:
            torch.Tensor: A binary mask indicating which weights should be pruned.

        """
        mask = torch.ones_like(weights, dtype=torch.bool)  # Initialize mask as all True

        if self.min_threshold is not None:
            mask = mask & (torch.abs(weights) > self.min_threshold)
        
        if self.max_threshold is not None:
            mask = mask & (torch.abs(weights) < self.max_threshold)
       
        return mask.float()  # Convert to float (1.0 for keep, 0.0 for prune)

