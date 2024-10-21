import torch
import torch.nn as nn
from criterion.magnitude_criterion import MagnitudeCriterion
from criterion.random_criterion import RandomCriterion
from context.global_pruning_context import GlobalPruningContext
from context.layer_pruning_context import LayerPruningContext
from criterion.percentile_criterion import PercentileCriterion
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


# Define your model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 26 * 26, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def test_magnitude_criterion():
    print("Testing Magnitude Criterion")

    # Initialize the model
    model = MyModel()

    # Define global pruning criteria
    magnitude_criterion1 = MagnitudeCriterion(min_threshold=0.01)
    magnitude_criterion2 = MagnitudeCriterion(max_threshold=1.0)

    # Initialize the GlobalPruningContext with multiple criteria
    global_pruner = GlobalPruningContext(model, criteria=[magnitude_criterion1, magnitude_criterion2])

    # 1) Total number of parameters before pruning
    print_total_params(model, label="Before pruning:")

    # 2) Total model size before pruning
    print_model_size_in_bytes(model, label="Before pruning:")

    # Apply global pruning
    global_pruner.prune()

    # Convert pruned model to sparse format
    pruned_sparse_model = sparsify_model(model)

    # 3) Total number of non-zero parameters after pruning (sparse model)
    print_sparse_model_info(pruned_sparse_model, label="After pruning (sparse):")
    print("\n\n")

def test_random_criterion():
    print("Testing Random Criterion")

    # Initialize the model
    model = MyModel()

    # Define global pruning criteria
    criterion1 = RandomCriterion(prune_ratio=0.5)

    # Initialize the GlobalPruningContext with multiple criteria
    global_pruner = GlobalPruningContext(model, criteria=[criterion1])

    # 1) Total number of parameters before pruning
    print_total_params(model, label="Before pruning:")

    # 2) Total model size before pruning
    print_model_size_in_bytes(model, label="Before pruning:")

    # Apply global pruning
    global_pruner.prune()

    # Convert pruned model to sparse format
    pruned_sparse_model = sparsify_model(model)

    # 3) Total number of non-zero parameters after pruning (sparse model)
    print_sparse_model_info(pruned_sparse_model, label="After pruning (sparse):")
    print("\n\n")

def test_percentile_criterion():
    print("Testing Percentile Criterion")

    # Initialize the model
    model = MyModel()

    # Define global pruning criteria
    criterion1 = PercentileCriterion(0.2,0.9)

    # Initialize the GlobalPruningContext with multiple criteria
    global_pruner = GlobalPruningContext(model, criteria=[criterion1])

    # 1) Total number of parameters before pruning
    print_total_params(model, label="Before pruning:")

    # 2) Total model size before pruning
    print_model_size_in_bytes(model, label="Before pruning:")

    # Apply global pruning
    global_pruner.prune()

    # Convert pruned model to sparse format
    pruned_sparse_model = sparsify_model(model)

    # 3) Total number of non-zero parameters after pruning (sparse model)
    print_sparse_model_info(pruned_sparse_model, label="After pruning (sparse):")
    print("\n\n")

def test_layer_pruning():
    print("TEsting layer pruning")
    # Initialize the model
    model = MyModel()

    layer_criteria = {
    model.conv1: [RandomCriterion(prune_ratio=0.9)],
    model.conv2: [RandomCriterion(prune_ratio=0.9),PercentileCriterion(0.2,0.9)]
    }

    # Print the number of non-zero parameters before pruning
    non_zero_before = count_non_zero_params(model)
    print(f"Non-zero parameters before pruning: {non_zero_before}")


    layer_pruner = LayerPruningContext(layer_criteria)
    layer_pruner.prune()

    # Print the number of non-zero parameters after pruning
    non_zero_after = count_non_zero_params(model)
    print(f"Non-zero parameters after pruning: {non_zero_after}")



if __name__ == '__main__':
    test_layer_pruning()
    # test_magnitude_criterion()
    # test_random_criterion()
    # test_percentile_criterion()
