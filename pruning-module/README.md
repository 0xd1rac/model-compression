# Pruning Module (Dummy)

This repository contains a dummy implementation of various model pruning techniques written to explore and get familiar with different ways to prune neural network layers. The module covers several popular pruning strategies, including magnitude-based pruning, random pruning, percentile-based pruning, and more. The overall objective is to learn how to effectively reduce model size while retaining its performance by selectively removing weights.

## Folder Structure


### `context/`

- **global_pruning_context.py**: Defines a pruning strategy that applies globally across the entire model, applying specific pruning criteria to all layers uniformly.
  
- **layer_pruning_context.py**: Allows for more fine-grained control by applying different pruning criteria to individual layers of the model. This context focuses on layer-specific pruning.

### `criterion/`

- **magnitude_criterion.py**: Implements magnitude-based pruning, which prunes weights below a specified threshold. This technique assumes that small weights contribute less to the overall performance of the model.

- **percentile_criterion.py**: Implements a percentile-based pruning strategy, pruning weights that fall outside a defined percentile range (e.g., pruning the bottom and top 10% of weights).

- **random_criterion.py**: Implements random pruning, where a percentage of weights are randomly selected and pruned. This approach is generally used for experimental purposes.

- **pruning_criterion.py**: Defines a base class (`PruningCriterion`) from which other pruning criteria inherit. This file serves as the foundation for creating new pruning techniques.

- **weights_manager.py**: Responsible for collecting the weights from a model, flattening them, and handling operations like mask application across layers.

- **helper.py**: Contains utility functions to assist in the pruning process, such as calculations and mask management.

## Pruning Techniques

The following techniques are implemented:

- **Global Pruning**: Prunes weights across the entire model uniformly based on a single criterion.
- **Layer-Wise Pruning**: Prunes each layer according to different criteria, providing more flexibility and finer control over the pruning process.
- **Magnitude-Based Pruning**: Removes weights whose absolute values fall below a certain threshold.
- **Percentile-Based Pruning**: Prunes weights outside of a certain percentile range.
- **Random Pruning**: Randomly prunes a percentage of weights, typically for experimental purposes.

## How to Use

### Example 1: Layer-Specific Pruning

```python
import torch.nn as nn
from criterion.magnitude_criterion import MagnitudeCriterion
from criterion.random_criterion import RandomCriterion
from context.layer_pruning_context import LayerPruningContext

# Example model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return x

# Initialize model
model = SimpleModel()

# Define layer-specific criteria
layer_criteria = {
    model.conv1: [MagnitudeCriterion(min_threshold=0.1), RandomCriterion(prune_ratio=0.2)],
    model.conv2: [PercentileCriterion(0.2, 0.9)]
}

# Apply pruning using LayerPruningContext
pruning_context = LayerPruningContext(layer_criteria)
pruning_context.prune()
```

### Example 2: Global Pruning
```python
import torch.nn as nn
from criterion.magnitude_criterion import MagnitudeCriterion
from context.global_pruning_context import GlobalPruningContext

# Example model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Initialize model
model = SimpleModel()

# Define global criteria (same criterion applied to all layers)
global_criteria = [
    MagnitudeCriterion(min_threshold=0.1)
]

# Apply global pruning
global_pruner = GlobalPruningContext(model, global_criteria)
global_pruner.prune()

```