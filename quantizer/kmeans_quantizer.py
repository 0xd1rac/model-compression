import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from sklearn.cluster import KMeans
import numpy as np

class KMeansQuantizer:
    def __init__(self, n_clusters:int=4):
        """
        Initialize the KMeansQuantizer.
        
        Parameters:
        - n_clusters: Number of clusters for K-Means quantization.
        """
        self.n_clusters = n_clusters
        self.quantizers = {}

    def fit(self, model):
        """Apply K-Means clustering to all layers with weights in the model"""
        for name, layer in model.named_modules():
            if hasattr(layer, 'weight') and layer.weight is not None:
                layer_weights = layer.weight.data.cpu().numpy()
                original_shape = layer_weights.shape
                weights_flat = layer_weights.flatten().reshape(-1,1)

                # Perform K-Means clustering 
                kmeans = KMeans(n_clusters=self.n_clusters)
                kmeans.fit(weights_flat)

                # Store quantizer information (centroids, labels) for the layer
                self.quantizers[name] = {
                    'centroids': kmeans.cluster_centers_.flatten(),
                    'labels': kmeans.labels_,
                    'original_shape': original_shape
                }

                # Quantize the weights by replacing them with centroids
                quantized_weights = kmeans.cluster_centers_[kmeans.labels_].reshape(original_shape)
                layer.weight.data = torch.tensor(quantized_weights, dtype=torch.float32).to(layer.weight.device)

    def fine_tune(self, model, lr:float=0.001):
        """ Fine-tune centroids using the gradients from backpropagation. """
        for name, layer in model.named_modules():
            if name in self.quantizers:
                grad_weights = layer.weight.grad.data.cpu().numpy().flatten()

                # Get the stored centroids and labels for this layer
                centroids = self.quantizers[name]['centroids']
                labels = self.quantizers[name]['labels']

                # Compute gradient for each centroid by averaging the gradients of the weights in each cluster
                grad_centroids = np.zeros_like(centroids)
                for i in range(len(centroids)):
                    grad_centroids[i] = grad_weights[labels == i].mean()

                # Update rule for centroids using Gradient Descent
                centroids -= lr * grad_centroids

                # Recompute weights using updated centroids 
                updated_weights = centroids[labels].reshape(self.quantizers[name]['original_shape'])
                layer.weight.data = torch.tensor(updated_weights, dtype=torch.float32).to(layer.weight.device)

    def get_quantized_weights(self, model):
        """ Return the quantized weights for all layers. """
        quantized_weights = {}
        for name, layer in model.named_modules():
            if name in self.quantizers:
                quantized_weights[name] = layer.weight.data
        return quantized_weights

    def get_centroids(self):
        """ Return the centroids (codebook) for all quantized layers. """
        centroids = {}
        for name, data in self.quantizers.items():
            centroids[name] = data['centroids']
        return centroids
    