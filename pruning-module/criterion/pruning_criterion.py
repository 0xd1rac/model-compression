from abc import ABC, abstractmethod
import torch.nn as nn
import torch

class PruningCriterion(ABC):
    @abstractmethod
    def get_mask_for_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the pruning strategy to a tensor and return a binary mask.
        
        Args:
            tensor (torch.Tensor): The tensor to prune.
            
        Returns:
            torch.Tensor: A binary mask indicating which weights to prune (0.0 for prune, 1.0 for keep).
        """
        pass

    @abstractmethod
    def get_mask_for_model(self, model: nn.Module) -> dict:
        """
        Apply the pruning strategy to the model and return the masks for each layer.
        
        Args:
            model (torch.nn.Module): The model to prune.
            
        Returns:
            dict: A dictionary with layer names as keys and binary masks (torch.Tensor) as values.
        """
        pass
