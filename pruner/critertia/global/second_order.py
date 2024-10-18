import torch
import torch.nn as nn
import torch.autograd as autograd

class SecondOrderCriterion:
    def __init__(self, prune_ratio: float):
        """
        Initialize the second-order pruning criterion.
        
        Args:
            prune_ratio (float): Fraction of weights to prune based on second-order information (e.g., 0.2 for 20%).
        """
        assert 0.0 <= prune_ratio <= 1.0, "prune_ratio must be between 0 and 1"
        self.prune_ratio = prune_ratio

    def compute_hessian_diag(self, model: nn.Module, loss: torch.Tensor) -> dict:
        """
        Compute the diagonal of the Hessian matrix for all weights in the model.
        
        Args:
            model (nn.Module): The neural network model.
            loss (torch.Tensor): The loss value used to compute the gradients.
        
        Returns:
            dict: A dictionary with layer names as keys and the diagonal of the Hessian as values.
        """
        hessian_diag = {}

        # First compute gradients of the loss w.r.t. weights
        grad_params = autograd.grad(loss, model.parameters(), create_graph=True)

        # Iterate over the gradients to compute the second derivative (Hessian diagonal)
        for i, param in enumerate(model.parameters()):
            if param.requires_grad:
                # Compute second derivative (diagonal of the Hessian)
                grad2 = autograd.grad(grad_params[i], param, retain_graph=True)[0]
                hessian_diag[param] = grad2.detach()  # Detach to avoid tracking further gradients

        return hessian_diag

    def apply(self, model: nn.Module, loss: torch.Tensor) -> dict:
        """
        Apply the second-order pruning criterion across all weights in the model.
        
        Args:
            model (nn.Module): The neural network model.
            loss (torch.Tensor): The loss function value (computed after forward pass).
        
        Returns:
            dict: A dictionary with layer names as keys and binary masks as values indicating which weights to keep.
        """
        hessian_diag = self.compute_hessian_diag(model, loss)
        all_hessian_values = []

        # Collect all second-order values (diagonal Hessian) for pruning
        for param in model.parameters():
            if param.requires_grad and param in hessian_diag:
                all_hessian_values.append(hessian_diag[param].flatten())

        # Concatenate all hessian values into a single tensor
        all_hessian_values = torch.cat(all_hessian_values)

        # Determine the threshold for pruning based on prune_ratio
        num_weights_to_prune = int(self.prune_ratio * all_hessian_values.numel())
        prune_threshold = torch.topk(all_hessian_values.abs(), num_weights_to_prune, largest=False)[0].max()

        # Create masks for each layer based on the second-order criterion
        masks = {}
        for param in model.parameters():
            if param.requires_grad and param in hessian_diag:
                # Prune weights with second-order values below the threshold
                mask = hessian_diag[param].abs() > prune_threshold
                masks[param] = mask.float()  # Convert to float for later multiplication with weights

        return masks
