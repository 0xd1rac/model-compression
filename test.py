from quantization.kmeans_quantization import KMeansQuantizer
import torch.optim as optim
import torch
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load pre-trained ResNet18 model
model = models.resnet18(pretrained=True)

# Load CIFAR-10 dataset (or a similar small dataset)
transform = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()])
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Define loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create an instance of KMeansQuantizer
quantizer = KMeansQuantizer(n_clusters=4)

# Function to extract the weights from a specific layer
def get_layer_weights(model, layer_name='conv1'):
    layer = dict(model.named_modules())[layer_name]
    return layer.weight.data.cpu().numpy()

# Plot the histograms for the weights before quantization, after quantization, and after fine-tuning
def plot_weight_distributions(before_quant, after_quant, after_finetune, layer_name='conv1', filename="weights_plot.png"):
    plt.figure(figsize=(12, 8))

    # Before quantization
    plt.subplot(1, 3, 1)
    plt.hist(before_quant.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title(f'{layer_name} Weights - Before Quantization')

    # After quantization (before fine-tuning)
    plt.subplot(1, 3, 2)
    plt.hist(after_quant.flatten(), bins=50, color='green', alpha=0.7)
    plt.title(f'{layer_name} Weights - After Quantization (Before Fine-tuning)')

    # After quantization (after fine-tuning)
    plt.subplot(1, 3, 3)
    plt.hist(after_finetune.flatten(), bins=50, color='red', alpha=0.7)
    plt.title(f'{layer_name} Weights - After Fine-tuning')

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved to {filename}")

def fine_tune_model(model, quantizer, testloader, n_epochs=5, lr=0.01):
    """ Fine-tune the quantized layers' centroids using backpropagation. """
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(n_epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(testloader, desc=f'Epoch {epoch + 1}/{n_epochs}', unit='batch'):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Fine-tune centroids based on layer gradients
            quantizer.fine_tune(model, lr=lr)

            # Perform SGD optimization on other model parameters
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(testloader)}")

# Step 1: Get weights before quantization
before_quant = get_layer_weights(model, layer_name='conv1')

# Step 2: Apply KMeans quantization to all layers with weights
quantizer.fit(model)

# Get the weights after quantization but before fine-tuning
after_quant = get_layer_weights(model, layer_name='conv1')

# Step 3: Fine-tune the model for a few epochs
fine_tune_model(model, quantizer, testloader, n_epochs=5)

# Step 4: Get the weights after fine-tuning
after_finetune = get_layer_weights(model, layer_name='conv1')

# Step 5: Plot the distributions and save to a file
plot_weight_distributions(before_quant, after_quant, after_finetune, layer_name='conv1', filename='weights_plot.png')
