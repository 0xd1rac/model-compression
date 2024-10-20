# model-compression

## Road Map

### Pruning 
#### Part 1: Learning about different pruning criterion
- [ ] Train a Resnet-50 on CIFAR-10 dataset and get a baseline accuracy
- [ ] Implement global pruning on trained Resnet-50 (Magnitude-based)
- [ ] Implement layer-wise pruning on trained Resnet-50 (Magnitude-based)
- [ ] Implement global pruning on trained Resnet-50 (Scaling-based)
- [ ] Implement layer-wise pruning on trained Resnet-50 (Scaling-based)
- [ ] Implement global pruning on trained Resnet-50 (Second-Order-based)
- [ ] Implement layer-wise pruning on trained Resnet-50 (Second-Order-based)
- [ ] Implement global pruning on trained Resnet-50 (Percentage-of Zero-based)
- [ ] Implement global layer-wise on trained Resnet-50 (Percentage-of Zero-based)
- [ ] Implement global pruning on trained Resnet-50 (Regression-based)
- [ ] Implement global layer-wise on trained Resnet-50 (Regression-based)

#### Part 2: Learning about different pruning schedules and finetuning
1. Set up the Environment 
- [ ] Train a baseline VGG16 model on CIFAR-10
- [ ] Select the amount of pruning you want to apply (e.g., prune 10%, 30%, or 50% of weights).
- [ ] Select a global or layerwise pruning criterion (one of the ones from Part 1)
- [ ] Apply One-shot pruning: Train the model fully, prune the chosen percentage of weights all at once, then fine-tune the pruned model for a few epochs.
- [ ] Apply Iteraitve pruning: Every ‘n’ epochs (e.g., every 10 epochs), prune a percentage of weights (e.g., 10%), retrain the pruned model, and repeat until the desired pruning percentage is reached.
- [ ] Gradual Pruning: Implement a pruning schedule where the amount of pruned weights increases over time (e.g., starting at 10%, increasing to 50% over 30 epochs).
- [ ] Fine-Tune the pruned Models: After applying pruning, fine-tune the models to restore accuracy, adjusting hyperparameters like learning rate if needed.
- [ ] Evaluate and Compare: Metrics - Accuracy, Training Time, Memory Usage (Number of parameters), Speed (Inference time)



### Quantization

### Pruning + Quanization

### Knowledge Distillation
