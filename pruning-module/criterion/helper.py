import torch
import torch.nn as nn
from typing import List, Tuple, Dict

# Function to collect model weights and return flattened weights and their mapping
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

# Function to calculate and print total number of parameters
def print_total_params(model, label=""):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{label} Total number of parameters: {total_params}")
    return total_params

# Function to calculate and print model size in MB for dense model
def print_model_size_in_bytes(model, label=""):
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"{label} Model size: {total_size / 1024 / 1024:.2f} MB")
    return total_size

# Function to print total number of non-zero parameters and size in MB for sparse model
def print_sparse_model_info(sparse_weights, label=""):
    total_sparse_params = sum(sparse_weight._nnz() for sparse_weight in sparse_weights.values())  # _nnz gives non-zero elements
    total_sparse_size = sum(sparse_weight._nnz() * sparse_weight.element_size() for sparse_weight in sparse_weights.values())  # Size in bytes
    
    print(f"{label} Total non-zero parameters: {total_sparse_params}")
    print(f"{label} Sparse model size: {total_sparse_size / 1024 / 1024:.2f} MB")
    return total_sparse_params, total_sparse_size

# Convert weights to sparse before saving for deployment
def sparsify_model(model):
    """
    Convert the weights of the model to sparse format and return them in a dictionary.
    """
    sparse_weights = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            sparse_weights[name] = param.data.to_sparse()  # Convert dense weight to sparse
        
    return sparse_weights  # Return the sparse weights separately
