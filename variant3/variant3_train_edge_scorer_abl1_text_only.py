"""
Train EdgeScorer Ablation 1: Text Embeddings Only

Ablation study to verify that the dual model learns from graph embeddings
and hop information. This version uses ONLY text embeddings.

Key differences from dual:
- No graph embeddings
- No hop information
- Only 4 components instead of 5
"""

import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import time
import os
from dotenv import load_dotenv
import wandb

# Load environment variables from .env file
load_dotenv()

from qa_system.config import Config
from variant3.variant3_edge_scorer_abl1_text_only import EdgeScorerTextOnly, count_parameters


class EdgeScorerTextOnlyDataset(Dataset):
    """PyTorch dataset for text-only ablation"""

    def __init__(
        self,
        samples: list,
        text_embeddings_cache: dict
    ):
        """
        Args:
            samples: Training samples from variant3_create_training_data.py
            text_embeddings_cache: Dict text -> 1536-dim embedding
        """
        self.samples = samples
        self.text_embeddings_cache = text_embeddings_cache

        # Precompute zero vector for unknowns
        self.zero_text = np.zeros(1536, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Extract sample fields
        question_text = sample['question_text']
        node_id = sample['node_id']
        relation = sample['edge_relation']
        target_id = sample['edge_target']
        label = sample['label']

        # Get text embeddings
        q_text = self.text_embeddings_cache.get(question_text, self.zero_text)
        n_text = self.text_embeddings_cache.get(node_id, self.zero_text)
        e_text = self.text_embeddings_cache.get(relation, self.zero_text)
        t_text = self.text_embeddings_cache.get(target_id, self.zero_text)

        return {
            'question_text_emb': torch.tensor(q_text, dtype=torch.float32),
            'node_text_emb': torch.tensor(n_text, dtype=torch.float32),
            'edge_text_emb': torch.tensor(e_text, dtype=torch.float32),
            'target_text_emb': torch.tensor(t_text, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }


def train_epoch(model, dataloader, optimizer, criterion, device, log_wandb=False, epoch=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_labels = []
    all_preds = []
    all_probs = []

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        question_text = batch['question_text_emb'].to(device)
        node_text = batch['node_text_emb'].to(device)
        edge_text = batch['edge_text_emb'].to(device)
        target_text = batch['target_text_emb'].to(device)
        labels = batch['label'].to(device).unsqueeze(1)

        # Forward
        optimizer.zero_grad()
        logits = model(question_text, node_text, edge_text, target_text)
        loss = criterion(logits, labels)

        # Backward
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Log batch loss to WandB
        if log_wandb and batch_idx % 100 == 0:
            wandb.log({
                'train/batch_loss': loss.item(),
                'train/batch': epoch * len(dataloader) + batch_idx
            })

        # Track predictions
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    # Compute epoch metrics
    print("\n   Computing metrics...", end=" ", flush=True)
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # AUC-ROC
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    print("Done!")

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


def evaluate(model, dataloader, criterion, device, return_predictions=False):
    """Evaluate on validation set"""
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in pbar:
            question_text = batch['question_text_emb'].to(device)
            node_text = batch['node_text_emb'].to(device)
            edge_text = batch['edge_text_emb'].to(device)
            target_text = batch['target_text_emb'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)

            logits = model(question_text, node_text, edge_text, target_text)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    # Overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # AUC-ROC
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    results = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

    if return_predictions:
        results['predictions'] = {
            'labels': np.array(all_labels),
            'preds': np.array(all_preds),
            'probs': np.array(all_probs)
        }

    return results


def main():
    print("=" * 80)
    print("Variant 3 Ablation 1: Text Embeddings Only (No Hops, No Graph)")
    print("=" * 80)

    # Hyperparameters (same as dual for fair comparison)
    BATCH_SIZE = 2048
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    PATIENCE = 10
    VAL_SPLIT = 0.2
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # WandB configuration
    USE_WANDB = True
    WANDB_PROJECT = "big-data-qa-system"
    WANDB_RUN_NAME = "variant3-ablation1-text-only"

    print(f"\n[Configuration]")
    print(f"   Device: {DEVICE}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Max epochs: {NUM_EPOCHS}")
    print(f"   Early stopping patience: {PATIENCE}")
    print(f"   WandB logging: {USE_WANDB}")

    # Initialize Weights & Biases
    if USE_WANDB:
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            config={
                "architecture": "EdgeScorerTextOnly",
                "ablation": "text_only",
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "max_epochs": NUM_EPOCHS,
                "patience": PATIENCE,
                "val_split": VAL_SPLIT,
                "text_embedding_dim": 1536,
                "hidden_dim": 512,
                "device": str(DEVICE),
                "optimizer": "AdamW",
                "loss": "BCEWithLogitsLoss",
            }
        )
        print(f"   WandB run: {wandb.run.name}")

    # Load training data
    print("\n[1/4] Loading training data...")
    training_data_path = Config.BASE_DIR / "data/variant3/variant3_training_data.pkl"
    with open(training_data_path, 'rb') as f:
        all_samples = pickle.load(f)
    print(f"   Total samples: {len(all_samples):,}")

    # Load text embeddings
    print("\n[2/4] Loading text embeddings...")
    emb_dir = Config.EMBEDDINGS_DIR

    with open(emb_dir / "variant3_text_embeddings_cache.pkl", 'rb') as f:
        text_embeddings_cache = pickle.load(f)

    print(f"   Text embeddings cache: {len(text_embeddings_cache):,} unique texts")

    # Train/val split by question ID (same as dual)
    print("\n[3/4] Creating train/val split...")
    unique_questions = list(set(s['question_id'] for s in all_samples))
    np.random.seed(42)
    np.random.shuffle(unique_questions)

    split_idx = int(len(unique_questions) * (1 - VAL_SPLIT))
    train_questions = set(unique_questions[:split_idx])
    val_questions = set(unique_questions[split_idx:])

    train_samples = [s for s in all_samples if s['question_id'] in train_questions]
    val_samples = [s for s in all_samples if s['question_id'] in val_questions]

    print(f"   Train samples: {len(train_samples):,}")
    print(f"   Val samples: {len(val_samples):,}")

    # Compute class weights
    train_labels = [s['label'] for s in train_samples]
    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    pos_weight = neg_count / pos_count

    print(f"   Positive samples: {pos_count:,}")
    print(f"   Negative samples: {neg_count:,}")
    print(f"   Pos weight: {pos_weight:.2f}")

    # Create datasets
    print("\n[4/4] Creating PyTorch datasets...")
    train_dataset = EdgeScorerTextOnlyDataset(
        train_samples,
        text_embeddings_cache
    )
    val_dataset = EdgeScorerTextOnlyDataset(
        val_samples,
        text_embeddings_cache
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    # Create model
    print("\n[5/5] Creating model...")
    model = EdgeScorerTextOnly(
        text_dim=1536,
        hidden_dim=512,
        dropout=0.3
    ).to(DEVICE)

    print(f"   Parameters: {count_parameters(model):,}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(DEVICE))
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    print(f"   Optimizer: AdamW (lr={LEARNING_RATE}, weight_decay=1e-5)")
    print(f"   Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)")

    # Training loop
    print("\n[Training...]")
    best_f1 = 0
    patience_counter = 0
    training_history = []

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, DEVICE,
                                     log_wandb=USE_WANDB, epoch=epoch)

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)

        # Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Print metrics
        print(f"   Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f} "
              f"(P: {train_metrics['precision']:.4f}, R: {train_metrics['recall']:.4f}), "
              f"AUC: {train_metrics['auc']:.4f}")
        print(f"   Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f} "
              f"(P: {val_metrics['precision']:.4f}, R: {val_metrics['recall']:.4f}), "
              f"AUC: {val_metrics['auc']:.4f}")
        print(f"   LR: {current_lr:.6f}")

        # Log to WandB
        if USE_WANDB:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'],
                'train/precision': train_metrics['precision'],
                'train/recall': train_metrics['recall'],
                'train/f1': train_metrics['f1'],
                'train/auc': train_metrics['auc'],
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'val/precision': val_metrics['precision'],
                'val/recall': val_metrics['recall'],
                'val/f1': val_metrics['f1'],
                'val/auc': val_metrics['auc'],
                'learning_rate': optimizer.param_groups[0]['lr'],
            })

        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        # Early stopping
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0

            # Save best model
            checkpoint_path = Config.BASE_DIR / "models" / "variant3_edge_scorer_abl1_text_only_best.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics
            }, checkpoint_path)
            print(f"   ✓ Best model saved (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n   Early stopping triggered (patience={PATIENCE})")
                break

    # Save training history
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    print(f"\nBest validation F1: {best_f1:.4f}")

    # Save as JSON
    models_dir = Config.BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    history_path = models_dir / "variant3_training_history_abl1_text_only.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"Training history (JSON) saved: {history_path}")

    # Save as CSV
    csv_rows = []
    for entry in training_history:
        epoch = entry['epoch']
        row = {
            'epoch': epoch,
            'learning_rate': entry['learning_rate'],
            'train_loss': entry['train_metrics']['loss'],
            'train_accuracy': entry['train_metrics']['accuracy'],
            'train_precision': entry['train_metrics']['precision'],
            'train_recall': entry['train_metrics']['recall'],
            'train_f1': entry['train_metrics']['f1'],
            'train_auc': entry['train_metrics']['auc'],
            'val_loss': entry['val_metrics']['loss'],
            'val_accuracy': entry['val_metrics']['accuracy'],
            'val_precision': entry['val_metrics']['precision'],
            'val_recall': entry['val_metrics']['recall'],
            'val_f1': entry['val_metrics']['f1'],
            'val_auc': entry['val_metrics']['auc'],
        }
        csv_rows.append(row)

    df = pd.DataFrame(csv_rows)
    csv_path = models_dir / "variant3_training_history_abl1_text_only.csv"
    df.to_csv(csv_path, index=False)
    print(f"Training history (CSV) saved: {csv_path}")

    # Save summary statistics
    summary_stats = {
        'ablation': 'text_only',
        'best_epoch': training_history[np.argmax([e['val_metrics']['f1'] for e in training_history])]['epoch'],
        'best_val_f1': best_f1,
        'best_val_accuracy': max([e['val_metrics']['accuracy'] for e in training_history]),
        'best_val_precision': max([e['val_metrics']['precision'] for e in training_history]),
        'best_val_recall': max([e['val_metrics']['recall'] for e in training_history]),
        'best_val_auc': max([e['val_metrics']['auc'] for e in training_history]),
        'final_train_loss': training_history[-1]['train_metrics']['loss'],
        'final_val_loss': training_history[-1]['val_metrics']['loss'],
        'total_epochs_trained': len(training_history),
        'model_parameters': count_parameters(model),
    }

    summary_path = models_dir / "variant3_training_summary_abl1_text_only.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"Training summary saved: {summary_path}")

    # Final validation analysis
    print("\n[Final Validation Analysis]")
    print("Loading best model for final evaluation...")

    checkpoint_path = models_dir / "variant3_edge_scorer_abl1_text_only_best.pt"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"   Best model loaded from epoch {checkpoint['epoch']}")

    # Get detailed predictions
    final_val_metrics = evaluate(model, val_loader, criterion, DEVICE, return_predictions=True)
    preds_data = final_val_metrics['predictions']

    # Confusion matrix
    cm = confusion_matrix(preds_data['labels'], preds_data['preds'])
    print(f"\n   Confusion Matrix:")
    print(f"      TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"      FN={cm[1,0]:,}  TP={cm[1,1]:,}")

    # Classification report
    class_report = classification_report(
        preds_data['labels'],
        preds_data['preds'],
        target_names=['Negative', 'Positive'],
        output_dict=True
    )

    # Save detailed validation results
    val_results = {
        'ablation': 'text_only',
        'confusion_matrix': {
            'TN': int(cm[0, 0]),
            'FP': int(cm[0, 1]),
            'FN': int(cm[1, 0]),
            'TP': int(cm[1, 1])
        },
        'classification_report': class_report,
        'overall_metrics': {
            'accuracy': final_val_metrics['accuracy'],
            'precision': final_val_metrics['precision'],
            'recall': final_val_metrics['recall'],
            'f1': final_val_metrics['f1'],
            'auc': final_val_metrics['auc']
        }
    }

    val_results_path = models_dir / "variant3_validation_results_abl1_text_only.json"
    with open(val_results_path, 'w') as f:
        json.dump(val_results, f, indent=2)
    print(f"\n   Validation results saved: {val_results_path}")

    # Log final summary to WandB
    if USE_WANDB:
        wandb.log({
            'best_val_f1': best_f1,
            'best_epoch': summary_stats['best_epoch'],
            'total_epochs': summary_stats['total_epochs_trained'],
            'confusion_matrix': wandb.plot.confusion_matrix(
                y_true=preds_data['labels'].flatten().tolist(),
                preds=preds_data['preds'].flatten().tolist(),
                class_names=['Negative', 'Positive']
            )
        })
        wandb.finish()
        print("\n   WandB run finished")

    print("\n" + "=" * 80)
    print("Ablation 1 (Text Only) complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
