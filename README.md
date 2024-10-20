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
  ** 1. Set up the Environment **
  - [ ] Train a baseline VGG16 model on CIFAR-10
  - [ ] Set up your dataset (e.g., CIFAR-10 or CIFAR-100).
  - [ ] Download or implement the Resnet50 Model
        
  ** 2. Train the Baseline Model **
  - [ ] Train the Resnet Model Until Convergence
  - [ ] Record the baseline performance metrics (accuracy, number of parameters, inferencen time)

  ** 3. Choose Pruning Percentages ** 
  - [ ] Decide on pruning percentages to try (e.g., 10%, 30%, 50% of weights).
  - [ ] Prepare the code to prune based on magnitude-based pruning (remove smallest weights by magnitude).

  ** 4. Implement Pruning Schedules ** 
  One-Shot Pruning
  - [ ] After training the full model, prune a fixed percentage of weights in one go (e.g., 30%).
  - [ ] Fine-tune the pruned model for a few epochs and record final accuracy and performance metrics.

 Iterative Pruning
 - [ ] Set up a loop that prunes a percentage of weights (e.g., 10%) every ‘n’ epochs (e.g., every 10 epochs).
 - [ ] Retrain the model between each pruning iteration to allow for recovery.
 - [ ] Repeat until the desired pruning percentage is reached (e.g., 50% total pruning).
 - [ ] Record accuracy and performance after each iteration.











### Quantization

### Pruning + Quanization

### Knowledge Distillation
