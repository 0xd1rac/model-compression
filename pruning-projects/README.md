## Project 1: Magnitude-based Pruning
  **1. Set up the Environment**
  - [ ] Set up your dataset (e.g., CIFAR-10 or CIFAR-100).
  - [ ] Download or implement the Resnet50 Model
        
  **2. Train the Baseline Model**
  - [ ] Train the Resnet Model Until Convergence
  - [ ] Record the baseline performance metrics (accuracy, number of parameters, inference time)

  **3. Implementing basic pruning techniques**
  - [ ] Pruning Strategy: Remove weights closest to zero (both globally and layer-wise).
  - [ ] Implement global magnitude-based pruning: prune the lowest magnitude weights across the entire network.
  - [ ] Implement layer-wise magnitude-based pruning: prune the lowest magnitude weights in each layer separately.
        
 **4. Fine-Tuning** 
  - [ ] Fine-the pruned models for a few epochs with a reduced learning rate.

 **5. Evaluation**
  - [ ] Measure the trade-off between the number of pruned weights, the model size, and the final accuracy after fine-tuning.


## Project 2: Structured vs Unstructured Pruning 
  **1. Set up the Environment**
  - [ ] Set up your dataset (e.g., CIFAR-10 or CIFAR-100).
  - [ ] Download or implement the Resnet50 Model
        
 **2. Train the Baseline Model**
  - [ ] Train the Resnet Model Until Convergence
  - [ ] Record the baseline performance metrics (accuracy, number of parameters, inference time)

 **3. Pruning**
  - [ ] Unstructured: prune individual weights based on their magnitude.
  - [ ] Structured: prune entire filters/channels based on the sum or average of their weights.

 **4. Fine-Tuning** 
  - [ ] Fine-the pruned models for a few epochs with a reduced learning rate.

 **5. Evaluation**
 - [ ] Measure the trade-off between the number of pruned weights, the model size, and the final accuracy after fine-tuning.


## Project 3: Sensitivity-based Pruning
  **1. Set up the Environment**
  - [ ] Set up your dataset (e.g., CIFAR-10 or CIFAR-100).
  - [ ] Download or implement the Resnet50 Model
        
  **2. Train the Baseline Model**
  - [ ] Train the Resnet Model Until Convergence
  - [ ] Record the baseline performance metrics (accuracy, number of parameters, inference time)\

  **3. Compute Sensitivity Scores**
  - [ ] Compute sensitivity scores using gradient norms or saliency maps for each weight/filter.
  - [ ] Rank weights/filters based on their sensitivity scores (least important first).

  **4. Compute Sensitivity Scores**
  - [ ] Prune the least sensitive weights/filters (e.g., prune the bottom 10-30%).
  - [ ] Implement both structured (filters) and unstructured (weights) pruning based on sensitivity scores.
        
  **5. Fine-Tuning** 
  - [ ] Fine-the pruned models for a few epochs with a reduced learning rate.

  **6. Evaluation**
  - [ ] Measure the trade-off between the number of pruned weights, the model size, and the final accuracy after fine-tuning.

 **7. Visualization** 
  - [ ]  Plot sensitivity maps or gradient norms for each layer.


## Project 4: Pruning Scheduling and Finetuning
  **1. Set up the Environment**
  - [ ] Set up your dataset (e.g., CIFAR-10 or CIFAR-100).
  - [ ] Download or implement the Resnet50 Model
        
  **2. Train the Baseline Model**
  - [ ] Train the Resnet Model Until Convergence
  - [ ] Record the baseline performance metrics (accuracy, number of parameters, inference time)

  **3. Choose Pruning Percentages** 
  - [ ] Decide on pruning percentages to try (e.g., 10%, 30%, 50% of weights).
  - [ ] Prepare the code to prune based on magnitude-based pruning (remove smallest weights by magnitude).

  **4. Implement Pruning Schedules** 
  
  **One-Shot Pruning**
  - [ ] After training the full model, prune a fixed percentage of weights in one go (e.g., 30%).
  - [ ] Fine-tune the pruned model for a few epochs and record final accuracy and performance metrics.


**Iterative Pruning**
 - [ ] Set up a loop that prunes a percentage of weights (e.g., 10%) every ‘n’ epochs (e.g., every 10 epochs).
 - [ ] Retrain the model between each pruning iteration to allow for recovery.
 - [ ] Repeat until the desired pruning percentage is reached (e.g., 50% total pruning).
 - [ ] Record accuracy and performance after each iteration.


**Gradual Pruning**
- [ ] Implement a pruning schedule that gradually increases the amount of pruning over time (e.g., start pruning 10% and gradually increase to 50% over 20-30 epochs).
- [ ] Fine-tune after each pruning step as the schedule progresses.
- [ ] Record accuracy and performance metrics throughout the schedule.


**Fine-Tune the Pruned Models**
- [ ] After pruning in each schedule (one-shot, iterative, gradual), fine-tune the pruned model to recover lost accuracy.
- [ ] Adjust hyperparameters like learning rate if needed during fine-tuning.

**Evaluate and Compare Results** 
- [ ] Compare the results of all pruning schedules against the baseline model:
    - Final Accuracy
    - Training Time
    - Model Size (number of parameters)
    - Inference Speed

 - [ ] Visualize the results using graphs (training loss, accuracy, parameters pruned).
