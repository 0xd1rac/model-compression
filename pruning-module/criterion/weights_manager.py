import torch
import torch.nn as nn
from typing import List, Tuple, Dict

class WeightsManager:
    @staticmethod
    def collect_model_weights(model: nn.Module) -> Tuple[torch.Tensor, List[Tuple[nn.Module, int]]]:
        """
        Collect and flatten all the weights from the model into a single tensor.
        Also, keep track of the number of parameters for each layer (mapping).

        Args:
            model (nn.Module): The neural network model to collect weights from.

        Returns:
            Tuple[torch.Tensor, List[Tuple[nn.Module, int]]]:
            - A concatenated tensor of all model weights.
            - A list of tuples where each tuple contains:
            (layer, number of parameters in that layer).
        """
        all_weights = []
        param_mapping = []

        # Iterate over the model layers and collect their weights
        for module in model.modules():
            if hasattr(module, 'weight') and isinstance(module.weight, torch.nn.Parameter):
                weight_flat = module.weight.data.flatten()  # Flatten the weights
                all_weights.append(weight_flat)
                param_mapping.append((module, weight_flat.numel()))  # Store the module and its num of params

        # Concatenate all flattened weights into a single tensor
        all_weights = torch.cat(all_weights)

        return all_weights, param_mapping

    @staticmethod
    def build_masks(tensor_mask: torch.Tensor, param_mapping: Tuple[torch.Tensor, List[Tuple[nn.Module, int]]]) -> Dict[nn.Module, torch.Tensor]:
        module_to_mask = {}
        current_pos = 0
        
        # Iterate through each module and its corresponding number of parameters
        for module, num_params in param_mapping:
            # Get the slice of the mask for the current layer
            layer_mask = tensor_mask[current_pos:current_pos + num_params]
            
            # Reshape the mask to match the shape of the current layer's weight
            layer_mask = layer_mask.view_as(module.weight)

             # Store the mask in the dictionary using the module's name or id as the key
            module_to_mask[module] = layer_mask

            # Update the current position to move to the next set of weights
            current_pos += num_params

        return module_to_mask
