# Variant 3 Training Metrics Tracking

## Overview

The training script now tracks comprehensive metrics suitable for research papers, with automatic logging to both **Weights & Biases (wandb)** and **local files**.

## Tracked Metrics

### Training Metrics (per epoch)
- **Loss**: Binary cross-entropy loss
- **Accuracy**: Overall classification accuracy
- **Precision**: Positive class precision
- **Recall**: Positive class recall (sensitivity)
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve

### Validation Metrics (per epoch)
- **Loss**: Binary cross-entropy loss
- **Accuracy**: Overall classification accuracy
- **Precision**: Positive class precision
- **Recall**: Positive class recall
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve

### Per-Hop Validation Metrics (per epoch)
For each hop (1, 2, 3):
- **Accuracy**: Hop-specific classification accuracy
- **Precision**: Hop-specific precision
- **Recall**: Hop-specific recall
- **F1 Score**: Hop-specific F1
- **AUC-ROC**: Hop-specific AUC
- **Num Samples**: Number of samples at this hop level

### Additional Tracking
- **Learning Rate**: Current optimizer learning rate (with ReduceLROnPlateau scheduling)
- **Epoch**: Current training epoch
- **Best F1**: Best validation F1 score achieved
- **Best Epoch**: Epoch where best F1 was achieved

## Output Files

All files are saved in the `models/` directory:

### 1. Validation Set Metadata: `variant3_validation_metadata.json`
Information about the validation split and composition.

**Format:**
```json
{
  "num_questions": 55247,
  "num_samples": 1138381,
  "pos_samples": 113812,
  "neg_samples": 1024569,
  "pos_ratio": 0.0999,
  "hop_distribution": {
    "0": 456234,
    "1": 512456,
    "2": 169691
  },
  "question_ids": [0, 1, 2, ...],
  "split_ratio": 0.2,
  "random_seed": 42
}
```

This is crucial for:
- **Reproducibility**: Exact validation questions used
- **Data distribution**: Class balance and hop distribution
- **Paper reporting**: Validation set characteristics

### 2. JSON Format: `variant3_training_history_dual.json`
Complete training history with all metrics per epoch. Includes nested structure for hop metrics.

**Format:**
```json
[
  {
    "epoch": 1,
    "learning_rate": 0.001,
    "train_metrics": {
      "loss": 0.1234,
      "accuracy": 0.8567,
      "precision": 0.8123,
      "recall": 0.7890,
      "f1": 0.8004,
      "auc": 0.9012
    },
    "val_metrics": {
      "loss": 0.1456,
      "accuracy": 0.8234,
      "precision": 0.7890,
      "recall": 0.7567,
      "f1": 0.7725,
      "auc": 0.8834,
      "hop_metrics": {
        "0": {
          "accuracy": 0.8912,
          "precision": 0.8567,
          "recall": 0.8234,
          "f1": 0.8397,
          "auc": 0.9234,
          "num_samples": 12345
        },
        ...
      }
    }
  },
  ...
]
```

### 3. CSV Format: `variant3_training_history_dual.csv`
Flattened format for easy plotting and analysis in Excel/Python/R.

**Columns:**
- `epoch`
- `learning_rate`
- `train_loss`, `train_accuracy`, `train_precision`, `train_recall`, `train_f1`, `train_auc`
- `val_loss`, `val_accuracy`, `val_precision`, `val_recall`, `val_f1`, `val_auc`
- `val_hop1_accuracy`, `val_hop1_precision`, `val_hop1_recall`, `val_hop1_f1`, `val_hop1_auc`, `val_hop1_num_samples`
- `val_hop2_accuracy`, `val_hop2_precision`, `val_hop2_recall`, `val_hop2_f1`, `val_hop2_auc`, `val_hop2_num_samples`
- `val_hop3_accuracy`, `val_hop3_precision`, `val_hop3_recall`, `val_hop3_f1`, `val_hop3_auc`, `val_hop3_num_samples`

### 4. Summary Statistics: `variant3_training_summary.json`
Key statistics for the paper's results section.

**Contents:**
```json
{
  "best_epoch": 42,
  "best_val_f1": 0.8567,
  "best_val_accuracy": 0.8912,
  "best_val_precision": 0.8734,
  "best_val_recall": 0.8401,
  "best_val_auc": 0.9234,
  "final_train_loss": 0.0987,
  "final_val_loss": 0.1234,
  "total_epochs_trained": 45,
  "model_parameters": 6912345
}
```

### 5. Validation Results: `variant3_validation_results.json`
Detailed analysis of the best model on validation set.

**Contents:**
```json
{
  "confusion_matrix": {
    "TN": 912456,
    "FP": 112113,
    "FN": 12345,
    "TP": 101467
  },
  "classification_report": {
    "Negative": {
      "precision": 0.8906,
      "recall": 0.8906,
      "f1-score": 0.8906,
      "support": 1024569
    },
    "Positive": {
      "precision": 0.4753,
      "recall": 0.8915,
      "f1-score": 0.6201,
      "support": 113812
    },
    "accuracy": 0.8906,
    "macro avg": {...},
    "weighted avg": {...}
  },
  "overall_metrics": {
    "accuracy": 0.8906,
    "precision": 0.8234,
    "recall": 0.8915,
    "f1": 0.8561,
    "auc": 0.9234
  },
  "hop_metrics": {
    "0": {...},
    "1": {...},
    "2": {...}
  }
}
```

**Use this for:**
- Confusion matrix for your paper
- Per-class performance analysis
- Understanding model errors (FP vs FN)

### 6. Validation Predictions: `variant3_validation_predictions.csv`
Individual predictions for every validation sample.

**Columns:**
- `label`: Ground truth (0 or 1)
- `prediction`: Model prediction (0 or 1)
- `probability`: Model's confidence score [0, 1]
- `hop`: Which hop this sample belongs to (0, 1, or 2)
- `correct`: Boolean indicating if prediction was correct

**Use this for:**
- Error analysis: Filter by `correct == False`
- Probability calibration analysis
- Per-hop error analysis
- Threshold optimization
- Plotting ROC curves

**Example analysis:**
```python
import pandas as pd

df = pd.read_csv('models/variant3_validation_predictions.csv')

# Find misclassified samples
errors = df[~df['correct']]
print(f"Total errors: {len(errors)}")

# Analyze false positives (predicted positive, actually negative)
false_positives = errors[errors['label'] == 0]
print(f"False positives: {len(false_positives)}")
print(f"Avg confidence: {false_positives['probability'].mean():.4f}")

# Per-hop error rates
for hop in [0, 1, 2]:
    hop_data = df[df['hop'] == hop]
    error_rate = (~hop_data['correct']).sum() / len(hop_data)
    print(f"Hop {hop+1} error rate: {error_rate:.4f}")
```

### 7. Model Checkpoint: `variant3_edge_scorer_dual_best.pt`
PyTorch checkpoint of the best model (highest validation F1).

**Contents:**
```python
{
  'epoch': 42,
  'model_state_dict': ...,
  'optimizer_state_dict': ...,
  'best_f1': 0.8567,
  'val_metrics': {...}
}
```

## Weights & Biases Integration

### Configuration

The script automatically logs to wandb with the following configuration:

- **Project**: `big-data-qa-system`
- **Run Name**: `variant3-edge-scorer-dual`
- **Tags**: Can be customized in the script

To **disable wandb** logging, set `USE_WANDB = False` in the training script.

### Wandb Logged Metrics

All metrics are logged per epoch with the following naming convention:

**Training metrics:**
- `train/loss`
- `train/accuracy`
- `train/precision`
- `train/recall`
- `train/f1`
- `train/auc`

**Validation metrics:**
- `val/loss`
- `val/accuracy`
- `val/precision`
- `val/recall`
- `val/f1`
- `val/auc`

**Per-hop validation metrics:**
- `val/hop_1_accuracy`, `val/hop_1_precision`, `val/hop_1_recall`, `val/hop_1_f1`, `val/hop_1_auc`
- `val/hop_2_accuracy`, `val/hop_2_precision`, `val/hop_2_recall`, `val/hop_2_f1`, `val/hop_2_auc`
- `val/hop_3_accuracy`, `val/hop_3_precision`, `val/hop_3_recall`, `val/hop_3_f1`, `val/hop_3_auc`

**Hyperparameters (logged as config):**
- `architecture`: EdgeScorerDual
- `batch_size`: 256
- `learning_rate`: 0.001
- `max_epochs`: 50
- `patience`: 5
- `val_split`: 0.2
- `text_embedding_dim`: 1536
- `graph_embedding_dim`: 256
- `hidden_dim`: 512
- `device`: cuda/cpu
- `optimizer`: AdamW
- `loss`: BCEWithLogitsLoss

### Accessing Wandb Dashboard

After training starts, you'll see a link to your wandb dashboard:

```
wandb: Run data is saved locally in wandb/...
wandb: Run `wandb offline` to turn off syncing
wandb: Syncing run variant3-edge-scorer-dual
wandb: ⭐️ View project at https://wandb.ai/<your-username>/big-data-qa-system
wandb: 🚀 View run at https://wandb.ai/<your-username>/big-data-qa-system/runs/<run-id>
```

## Plotting Results

### Python Example (using pandas)

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv('models/variant3_training_history_dual.csv')

# Plot training curves
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Loss
axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train')
axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val')
axes[0, 0].set_title('Loss')
axes[0, 0].legend()

# Accuracy
axes[0, 1].plot(df['epoch'], df['train_accuracy'], label='Train')
axes[0, 1].plot(df['epoch'], df['val_accuracy'], label='Val')
axes[0, 1].set_title('Accuracy')
axes[0, 1].legend()

# F1 Score
axes[0, 2].plot(df['epoch'], df['train_f1'], label='Train')
axes[0, 2].plot(df['epoch'], df['val_f1'], label='Val')
axes[0, 2].set_title('F1 Score')
axes[0, 2].legend()

# Precision
axes[1, 0].plot(df['epoch'], df['train_precision'], label='Train')
axes[1, 0].plot(df['epoch'], df['val_precision'], label='Val')
axes[1, 0].set_title('Precision')
axes[1, 0].legend()

# Recall
axes[1, 1].plot(df['epoch'], df['train_recall'], label='Train')
axes[1, 1].plot(df['epoch'], df['val_recall'], label='Val')
axes[1, 1].set_title('Recall')
axes[1, 1].legend()

# AUC
axes[1, 2].plot(df['epoch'], df['train_auc'], label='Train')
axes[1, 2].plot(df['epoch'], df['val_auc'], label='Val')
axes[1, 2].set_title('AUC-ROC')
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300)
plt.show()
```

### Per-Hop Analysis

```python
# Plot per-hop F1 scores
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['val_hop1_f1'], label='Hop 1')
plt.plot(df['epoch'], df['val_hop2_f1'], label='Hop 2')
plt.plot(df['epoch'], df['val_hop3_f1'], label='Hop 3')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('Validation F1 Score by Hop')
plt.legend()
plt.grid(True)
plt.savefig('per_hop_f1.png', dpi=300)
plt.show()
```

## For Your Paper

Use the summary statistics file for reporting results:

```python
import json

with open('models/variant3_training_summary.json') as f:
    stats = json.load(f)

print(f"Best Validation F1: {stats['best_val_f1']:.4f}")
print(f"Best Validation Accuracy: {stats['best_val_accuracy']:.4f}")
print(f"Best Validation AUC: {stats['best_val_auc']:.4f}")
print(f"Model Parameters: {stats['model_parameters']:,}")
print(f"Converged at Epoch: {stats['best_epoch']}")
```

Example output for paper:
> "The EdgeScorerDual model achieved a best validation F1 score of 0.8567
> (accuracy: 0.8912, AUC: 0.9234) at epoch 42, with 6.9M parameters."

## Early Stopping

The model uses early stopping based on validation F1 score:
- **Patience**: 5 epochs
- **Criterion**: No improvement in validation F1
- Best model is automatically saved

## Notes

1. All metrics use `zero_division=0` for sklearn to handle edge cases
2. AUC-ROC calculation is wrapped in try-except to handle cases with single class predictions
3. Per-hop metrics are only computed if samples exist for that hop
4. CSV format makes it easy to import into LaTeX tables or plotting tools
