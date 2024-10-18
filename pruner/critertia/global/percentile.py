import torch 
import numpy as np
import torch.nn as nn

class PercentileCriterion:
    def __init__(self, bottom_pct: float=None, top_pct: float=None):
        """
        Initialize the percentile pruning criterion for global pruning.
        
        Args:
            bottom_pct (float): The bottom percentile below which weights will be pruned (e.g., 0.1 for bottom 10%).
            top_pct (float): The top percentile above which weights will be pruned (e.g., 0.9 for top 10%).
        """
        if bottom_pct is not None:
            assert 0.0 <= bottom_pct <= 1.0, "bottom_pct must be between 0 and 1"
        
        if top_pct is not None:
            assert 0.0 <= top_pct <= 1.0, "top_pct must be between 0 and 1"
        
        if bottom_pct is not None and top_pct is not None:
            assert bottom_pct < top_pct, "bottom_pct must be less than top_pct"

        self.bottom_pct = bottom_pct
        self.top_pct = top_pct
    
    def collect_weights(self, model):
        all_weights = []

        # Collect all weights from the model
        for name, param in model.named_parameters():
            if 'weight' in name:
                all_weights.append(param.data.flatten())

        # Concatenate all weights into a single tensor
        all_weights = torch.cat(all_weights)
        return all_weights

    def apply(self, model: nn.Module) -> dict:
        """
        Apply the percentile-based pruning criterion across all weights in the model.
        
        Args:
            model (torch.nn.Module): The neural network model.
        
        Returns:
            dict: A dictionary with layer names as keys and binary masks as values.
        """
        all_weights = self.collect_weights(model)

        # Compute the threshold for the bottom and top percentile
        bottom_threshold, top_threshold = None, None

        # Calculate bottom and top thresholds if bottom_pct and top_pct are provided
        if self.bottom_pct is not None:
            bottom_threshold = np.percentile(all_weights.cpu().numpy(), self.bottom_pct * 100)

        if self.top_pct is not None:
            top_threshold = np.percentile(all_weights.cpu().numpy(), self.top_pct * 100)
       
        # Generate a mask for each layer based on the global percentile thresholds
        masks = {}
        for name, param in model.named_parameters():
            if 'weight' in name:
                mask = torch.ones_like(param.data, dtype=torch.bool)  # Initialize with all True

                # Apply bottom threshold if provided
                if bottom_threshold is not None:
                    mask = mask & (param.data >= bottom_threshold)

                if top_threshold is not None:
                    mask = mask & (param.data <= top_threshold)
                
                masks[name] = mask.float()  # Convert to float for later multiplication with weights

        return masks


