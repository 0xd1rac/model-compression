
import torch
import torch.nn as nn
import numpy as np 
from sklearn.linear_model import LinearRegression

class RegressionCriterion:
    def __init__(self, prune_ratio:float):
        """
        Initialize the regression-based pruning criterion for global pruning.
        
        Args:
            prune_ratio (float): Fraction of weights to prune globally based on residuals (e.g., 0.2 for 20%).
        """
        assert 0.0 <= prune_ratio <= 1.0, "prune_ratio must be between 0 and 1"
        self.prune_ratio = prune_ratio

    def collect_weights(self, model):
        all_weights = []
        layer_shapes = {}  # Store original shapes of the layers for reshaping masks later
        weight_tensors = {}


        # Collect all weights from the model
        for name, param in model.named_parameters():
            if 'weight' in name:
                flattened_weights = param.data.flatten()
                all_weights.append(flattened_weights)
                layer_shapes[name] = param.shape  # Store the shape of each layer
                weight_tensors[name] = flattened_weights  # Store the tensor itself

        # Concatenate all weights into a single tensor
        all_weights = torch.cat(all_weights)
        return all_weights, layer_shapes, weight_tensors
    
    def apply(self, model: nn.Module) -> dict:
        """
        Apply the regression-based pruning criterion globally across all weights in the model.
        
        Args:
            model (torch.nn.Module): The neural network model.
        
        Returns:
            dict: A dictionary with layer names as keys and binary masks as values.
        """
        
        all_weights, layer_shapes, weight_tensors = self.collect_weights(model)
        num_weights = len(all_weights)

        # Step 2: Fit a linear regression model to all global weights
        indices = np.arange(num_weights).reshape(-1, 1)  # Use weight indices as independent variable
        model = LinearRegression()
        model.fit(indices, all_weights)
        predicted_weights = model.predict(indices)

        # Step 3: Calculate residuals (difference between actual and predicted weights)
        residuals = np.abs(all_weights - predicted_weights)

         # Step 4: Sort residuals and determine the pruning threshold
        num_weights_to_prune = int(self.prune_ratio * num_weights)
        prune_threshold = np.partition(residuals, num_weights_to_prune)[num_weights_to_prune]

        # Step 5: Generate a global mask by setting weights with small residuals to 0
        global_mask = residuals > prune_threshold

        # Step 6: Split the global mask back into layer-specific masks
        masks = {}
        current_index = 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Get the number of elements in this layer
                num_elements = param.numel()

                # Extract the corresponding slice from the global mask
                layer_mask = global_mask[current_index:current_index + num_elements]

                # Reshape the mask back to the layer's original shape
                masks[name] = torch.tensor(layer_mask.reshape(layer_shapes[name]), dtype=torch.float32)

                # Move the current index forward
                current_index += num_elements

        return masks
