import torch
import torch.nn as nn

class MagnitudeCriterion:
    def __init__(self, min_threshold: float = None, max_threshold: float = None):
        """
        Global magnitude-based pruning zeros out weights smaller than min_threshold 
        and larger than max_threshold across all layers in the model.

        Args:  
            min_threshold (float): Threshold below which weights will be pruned.
            max_threshold (float): Threshold above which weights will be pruned.
        """
        assert min_threshold is not None or max_threshold is not None, "args, min_threshold OR max_threshold must be set."
        assert min_threshold < max_threshold, "min_threshold MUST be smaller than max_threshold."

        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def collect_weights(self, model):
        all_weights = []

        # Collect all weights from the model
        for name, param in model.named_parameters():
            if 'weight' in name:
                all_weights.append(param.data.flatten())

        # Concatenate all weights into a single tensor
        all_weights = torch.cat(all_weights)
        return all_weights

    def apply(self, model:nn.Module) -> dict:
        """
        Apply the global magnitude pruning criterion to all weights in the model.

        Args:
            model (torch.nn.Module): The neural network model.

        Returns:
            dict: A dictionary with layer names as keys and binary masks as values.
        """

        all_weights = self.collect_weights(model)

        # Determine min and max thresholds if not already set
        if self.min_threshold is None:
            self.min_threshold = all_weights.min().item()
        if self.max_threshold is None:
            self.max_threshold = all_weights.max().item()

        # Generate a mask for each layer based on the globa threshold
        masks = {}
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Apply global thresholds to prune weights
                mask = (torch.abs(param.data) > self.min_threshold) & (torch.abs(param.data) < self.max_threshold)
                masks[name] = mask.float()

        return masks