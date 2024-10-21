import torch
import torch.nn as nn
from typing import List, Dict
from criterion.pruning_criterion import PruningCriterion

class LayerPruningContext:
    def __init__(self, layer_to_criteria: Dict[nn.Module, List[PruningCriterion]]):
        """
        Initialize the context with a model and a list of global criteria.
        
        Args:
l           layer_to_criteria (Dict[nn.Module, List[PruningCriterion]]): A dictionary where the keys are layers (nn.Module)
                                                                      and the values are lists of PruningCriterion instances to apply to that layer.
        """
        self.layer_to_criteria = layer_to_criteria

    def prune(self) -> None:
        """
        Apply the layer-specific pruning strategies to the corresponding layers in the model.
        """
        combined_module_to_mask = {}  # To hold combined masks for each layer in layer_criteria

        with torch.no_grad():
            for module, criteria_list in self.layer_to_criteria.items():
                # Ensure the module has a weight attribute (e.g., conv, linear layers)
                if not hasattr(module, 'weight') or module.weight is None:
                    raise ValueError(f"Module {module} does not have weights to prune.")

                # Start with an initial mask of ones (keep all weights
                combined_mask = torch.ones_like(module.weight.data, dtype=torch.bool)

                for criterion in criteria_list:
                    mask = criterion.get_mask_for_tensor(module.weight.data)
                    combined_mask &= mask
                
                # Store the final combined mask for this layer
                combined_module_to_mask[module] = combined_mask
                
            # Step 2: Apply the final combined masks to the corresponding layers
            self.apply_masks(combined_module_to_mask)
    
    def apply_masks(self, module_to_mask:Dict[nn.Module, torch.Tensor]) -> None:
         """
        Apply the generated masks to the corresponding layers' weights.

        Args:
            module_to_mask (Dict[nn.Module, torch.Tensor]): A dictionary where keys are the layers (nn.Module) 
                                                            and values are the masks for each layer's weights.
        """
         
         for module, mask in module_to_mask.items():
            print(f"Pruning {module}")
             # Convert boolean mask to float (True -> 1.0, False -> 0.0)
            mask = mask.float()
            print(f"Number of zeroes: {mask.numel() - torch.count_nonzero(mask)}")

             # Apply the mask to the layer's weights
            with torch.no_grad():  # Ensure no gradient tracking during this operation
                module.weight.data *= mask  # Element-wise multiplication with the mask