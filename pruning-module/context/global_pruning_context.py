import torch
import torch.nn as nn
from typing import List, Dict
from criterion.pruning_criterion import PruningCriterion

class GlobalPruningContext:
    def __init__(self, model: nn.Module, criteria: List[PruningCriterion]):
        """
        Initialize the context with a model and a list of global criteria.
        
        Args:
            model (torch.nn.Module): The model to prune.
            criteria (List[MagnitudeCriterion]): A list of MagnitudeCriterion instances to apply globally.
        """
        self.model = model
        self.criteria = criteria

    def prune(self) -> None:
        """
        Apply the global pruning strategies to all layers in the model.
        """
        combined_module_to_mask = {}  # To hold the combined masks for each layer
        
        with torch.no_grad():
            # For each criterion, generate masks for the model's weights
            for criterion in self.criteria:
                module_to_mask = criterion.get_mask_for_model(self.model)  # Get a dictionary of masks
                
                # Combine masks for each layer (logical AND operation)
                for module, mask in module_to_mask.items():
                    if module not in combined_module_to_mask:
                        combined_module_to_mask[module] = mask  # First time for this layer
                    else:
                        combined_module_to_mask[module] &= mask  # Combine with the previous masks

            # Apply the final masks to the corresponding layers' weights
            self.apply_masks(combined_module_to_mask)

    def apply_masks(self, 
                    module_to_mask: Dict[nn.Module, torch.Tensor]
                    ) -> None:
        """
        Apply the generated masks to the corresponding layers' weights in the model.

        Args:
            module_to_mask (Dict[nn.Module, torch.Tensor]): A dictionary where keys are the layers (nn.Module) 
                                                        and values are the masks for each layer's weights.
        """
        for module in self.model.modules():
            if module in module_to_mask:
                mask = module_to_mask[module]
                mask = mask.float()
                
                # Apply the mask to the layer's weights
                with torch.no_grad():  # Ensure no gradient tracking during this operation
                    module.weight.data *= mask  # Element-wise multiplication with the mask
            