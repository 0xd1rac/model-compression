import torch

class RandomPruningCriterion:
    def __init__(self, prune_ratio: float):
        """
        Randomly prunes a fraction of weights.
        
        Args:
            prune_ratio (float): Fraction of weights to prune (e.g., 0.2 for 20%).
        """

        assert 0.0 <= prune_ratio <= 1.0, "prune_ratio must be between 0 and 1"
        self.prune_ratio = prune_ratio

    def apply(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Apply the random pruning criterion to the weights.

        Args:
            weights (torch.Tensor): The weights of the layer.

        Returns:
            torch.Tensor: A binary mask indicating which weights should be retained (1 for keep, 0 for prune).
        """
        # Flatten the weights for easier random sampling
        num_weights = weights.numel()
        num_to_prune = int(self.prune_ratio * num_weights)

        # Generate a mask where num_to_prune weights are randomly set to 0 (False)
        mask = torch.ones(num_weights, dtype=torch.bool)
        indices_to_prune = torch.randperm(num_weights)[:num_to_prune] # Random indices to prune
        mask[indices_to_prune] = False

        # Reshape the mask back to the original weight shape
        mask = mask.view_as(weights)

        return mask.float()

