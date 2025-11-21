"""
Train EdgeScorerDual with dual embeddings (text + graph) for all components.

Training procedure:
1. Load training data (question, node, edge, target_node, hop, label)
2. Load ALL cached embeddings:
   - Question text embeddings
   - Entity text embeddings
   - Relation text embeddings
   - TransE entity embeddings (graph)
   - TransE relation embeddings (graph)
3. Create PyTorch dataset with train/val split
4. Train with BCEWithLogitsLoss + pos_weight for class imbalance
5. Evaluate on validation set
6. Save best model checkpoint

Key features:
- Dual embeddings (text + graph) for richer representations
- Class balancing via pos_weight
- Early stopping based on validation F1
- Learning rate scheduling
- Per-hop accuracy tracking
"""

import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
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
from variant3.variant3_edge_scorer_dual import EdgeScorerDual, count_parameters


class EdgeScorerDualDataset(Dataset):
    """PyTorch dataset for EdgeScorerDual training with dual embeddings"""

    def __init__(
        self,
        samples: list,
        text_embeddings_cache: dict,
        entity_graph_embeddings: np.ndarray,
        entity2id: dict,
        relation_graph_embeddings: np.ndarray,
        relation2id: dict
    ):
        """
        Args:
            samples: Training samples from variant3_create_training_data.py
            text_embeddings_cache: Dict text -> 1536-dim embedding (unified cache for all texts)
            entity_graph_embeddings: TransE [num_entities, 256]
            entity2id: Dict entity_id -> TransE index
            relation_graph_embeddings: TransE [num_relations, 256]
            relation2id: Dict relation -> TransE index
        """
        self.samples = samples
        self.text_embeddings_cache = text_embeddings_cache
        self.entity_graph_embeddings = entity_graph_embeddings
        self.entity2id = entity2id
        self.relation_graph_embeddings = relation_graph_embeddings
        self.relation2id = relation2id

        # Precompute zero vectors for unknowns
        self.zero_text = np.zeros(1536, dtype=np.float32)
        self.zero_graph = np.zeros(256, dtype=np.float32)

        # Precompute question hop counts (infer from max hop per question)
        from collections import defaultdict
        question_max_hops = defaultdict(int)
        for sample in samples:
            qid = sample['question_id']
            hop = sample['hop']
            question_max_hops[qid] = max(question_max_hops[qid], hop)
        self.question_hop_counts = dict(question_max_hops)

        print(f"   Inferred question hop counts: "
              f"1-hop: {sum(1 for h in self.question_hop_counts.values() if h==1)}, "
              f"2-hop: {sum(1 for h in self.question_hop_counts.values() if h==2)}, "
              f"3-hop: {sum(1 for h in self.question_hop_counts.values() if h==3)}")

    def __len__(self):
        return len(self.samples)

    def _get_entity_embeddings(self, entity_id: str):
        """Get both text and graph embeddings for an entity"""
        # Text embedding from unified cache
        text_emb = self.text_embeddings_cache.get(entity_id, self.zero_text)

        # Graph embedding
        if entity_id in self.entity2id:
            idx = self.entity2id[entity_id]
            graph_emb = self.entity_graph_embeddings[idx]
        else:
            graph_emb = self.zero_graph

        return text_emb, graph_emb

    def _get_relation_embeddings(self, relation: str):
        """Get both text and graph embeddings for a relation"""
        # Text embedding from unified cache
        text_emb = self.text_embeddings_cache.get(relation, self.zero_text)

        # Graph embedding
        if relation in self.relation2id:
            idx = self.relation2id[relation]
            graph_emb = self.relation_graph_embeddings[idx]
        else:
            graph_emb = self.zero_graph

        return text_emb, graph_emb

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Extract sample fields
        question_id = sample['question_id']
        question_text = sample['question_text']
        node_id = sample['node_id']
        relation = sample['edge_relation']
        target_id = sample['edge_target']
        current_hop = sample['hop']  # 1, 2, or 3
        label = sample['label']

        # Get question hop count (1, 2, or 3)
        question_hop_count = self.question_hop_counts[question_id]

        # Create 6-bit one-hot hop context
        # First 3 bits: question hop count (1-hot)
        # Second 3 bits: current hop (1-hot)
        hop_context = np.zeros(6, dtype=np.float32)
        hop_context[question_hop_count - 1] = 1.0  # Question hop: 0-2 index
        hop_context[3 + current_hop - 1] = 1.0     # Current hop: 3-5 index

        # Question text embedding (no graph structure for questions)
        q_text = self.text_embeddings_cache.get(question_text, self.zero_text)

        # Node dual embeddings
        n_text, n_graph = self._get_entity_embeddings(node_id)

        # Edge dual embeddings
        e_text, e_graph = self._get_relation_embeddings(relation)

        # Target dual embeddings
        t_text, t_graph = self._get_entity_embeddings(target_id)

        return {
            'question_text_emb': torch.tensor(q_text, dtype=torch.float32),
            'node_text_emb': torch.tensor(n_text, dtype=torch.float32),
            'node_graph_emb': torch.tensor(n_graph, dtype=torch.float32),
            'edge_text_emb': torch.tensor(e_text, dtype=torch.float32),
            'edge_graph_emb': torch.tensor(e_graph, dtype=torch.float32),
            'target_text_emb': torch.tensor(t_text, dtype=torch.float32),
            'target_graph_emb': torch.tensor(t_graph, dtype=torch.float32),
            'hop_context': torch.tensor(hop_context, dtype=torch.float32),  # 6-dim
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
        node_graph = batch['node_graph_emb'].to(device)
        edge_text = batch['edge_text_emb'].to(device)
        edge_graph = batch['edge_graph_emb'].to(device)
        target_text = batch['target_text_emb'].to(device)
        target_graph = batch['target_graph_emb'].to(device)
        hop_context = batch['hop_context'].to(device)
        labels = batch['label'].to(device).unsqueeze(1)

        # Forward
        optimizer.zero_grad()
        logits = model(question_text, node_text, node_graph,
                       edge_text, edge_graph,
                       target_text, target_graph, hop_context)
        loss = criterion(logits, labels)

        # Backward
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

        # Update progress bar with current loss
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Log batch loss to WandB every 100 batches
        if log_wandb and batch_idx % 100 == 0:
            import wandb
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
    all_hops = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in pbar:
            question_text = batch['question_text_emb'].to(device)
            node_text = batch['node_text_emb'].to(device)
            node_graph = batch['node_graph_emb'].to(device)
            edge_text = batch['edge_text_emb'].to(device)
            edge_graph = batch['edge_graph_emb'].to(device)
            target_text = batch['target_text_emb'].to(device)
            target_graph = batch['target_graph_emb'].to(device)
            hop_context = batch['hop_context'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)

            logits = model(question_text, node_text, node_graph,
                           edge_text, edge_graph,
                           target_text, target_graph, hop_context)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

            # Extract current hop from hop_context (second 3 bits)
            current_hops = hop_context[:, 3:].argmax(dim=1).cpu().numpy()  # 0, 1, or 2
            all_hops.extend(current_hops)

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

    # Per-hop metrics
    hop_metrics = {}
    for hop_num in [0, 1, 2]:
        hop_mask = np.array(all_hops) == hop_num
        if hop_mask.sum() > 0:
            hop_labels = np.array(all_labels)[hop_mask]
            hop_preds = np.array(all_preds)[hop_mask]
            hop_probs = np.array(all_probs)[hop_mask]
            hop_acc = accuracy_score(hop_labels, hop_preds)
            hop_precision = precision_score(hop_labels, hop_preds, zero_division=0)
            hop_recall = recall_score(hop_labels, hop_preds, zero_division=0)
            hop_f1 = f1_score(hop_labels, hop_preds, zero_division=0)
            try:
                hop_auc = roc_auc_score(hop_labels, hop_probs)
            except:
                hop_auc = 0.0
            hop_metrics[hop_num] = {
                'accuracy': hop_acc,
                'precision': hop_precision,
                'recall': hop_recall,
                'f1': hop_f1,
                'auc': hop_auc,
                'num_samples': hop_mask.sum()
            }

    results = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'hop_metrics': hop_metrics
    }

    if return_predictions:
        results['predictions'] = {
            'labels': np.array(all_labels),
            'preds': np.array(all_preds),
            'probs': np.array(all_probs),
            'hops': np.array(all_hops)
        }

    return results


def main():
    print("=" * 80)
    print("Variant 3 Dual: Train EdgeScorer with Dual Embeddings")
    print("=" * 80)

    # Hyperparameters
    BATCH_SIZE = 2048  # Increased from 256 to utilize GPU memory (46GB A40)
    LEARNING_RATE = 1e-4  # Reduced from 1e-3 for better convergence with larger batches
    NUM_EPOCHS = 50
    PATIENCE = 10  # Increased from 5 to allow more time for convergence
    VAL_SPLIT = 0.2
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # WandB configuration
    USE_WANDB = True  # Set to False to disable wandb logging
    WANDB_PROJECT = "big-data-qa-system"
    WANDB_RUN_NAME = "variant3-edge-scorer-dual"

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
                "architecture": "EdgeScorerDual",
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "max_epochs": NUM_EPOCHS,
                "patience": PATIENCE,
                "val_split": VAL_SPLIT,
                "text_embedding_dim": 1536,
                "graph_embedding_dim": 256,
                "hidden_dim": 512,
                "device": str(DEVICE),
                "optimizer": "AdamW",
                "loss": "BCEWithLogitsLoss",
            }
        )
        print(f"   WandB run: {wandb.run.name}")

    # Load training data
    print("\n[1/6] Loading training data...")
    training_data_path = Config.BASE_DIR / "data/variant3/variant3_training_data.pkl"
    with open(training_data_path, 'rb') as f:
        all_samples = pickle.load(f)
    print(f"   Total samples: {len(all_samples):,}")

    # Load text embeddings (unified cache)
    print("\n[2/6] Loading text embeddings...")
    emb_dir = Config.EMBEDDINGS_DIR

    with open(emb_dir / "variant3_text_embeddings_cache.pkl", 'rb') as f:
        text_embeddings_cache = pickle.load(f)

    print(f"   Text embeddings cache: {len(text_embeddings_cache):,} unique texts")

    # Load graph embeddings (TransE)
    print("\n[3/6] Loading graph embeddings (TransE)...")
    entity_graph_embeddings = np.load(Config.TRANSE_ENTITY_PATH)
    relation_graph_embeddings = np.load(Config.TRANSE_RELATION_PATH)

    with open(Config.TRANSE_METADATA_PATH, 'r') as f:
        transe_metadata = json.load(f)

    # Load entity2id from node2id.json file (created during TransE training)
    node2id_path = Config.EMBEDDINGS_DIR / "node2id.json"
    if node2id_path.exists():
        print(f"   Loading entity mappings from {node2id_path.name}...")
        with open(node2id_path, 'r') as f:
            entity2id = json.load(f)
    else:
        raise FileNotFoundError(
            f"Entity mapping file not found: {node2id_path}\n"
            "This file should have been created during TransE training."
        )

    # Load relation2id from separate file (or fallback to metadata)
    relation2id_path = Config.EMBEDDINGS_DIR / "relation2id.json"
    if relation2id_path.exists():
        print(f"   Loading relation mappings from {relation2id_path.name}...")
        with open(relation2id_path, 'r') as f:
            relation2id = json.load(f)
    elif 'relation_to_id' in transe_metadata:
        print(f"   Loading relation mappings from TransE metadata...")
        relation2id = transe_metadata['relation_to_id']
    elif 'relation2id' in transe_metadata:
        print(f"   Loading relation mappings from TransE metadata...")
        relation2id = transe_metadata['relation2id']
    else:
        raise KeyError(
            "relation_to_id not found in TransE metadata or relation2id.json. "
            "Please check the TransE training output."
        )

    print(f"   Entity graph embeddings: {entity_graph_embeddings.shape}")
    print(f"   Relation graph embeddings: {relation_graph_embeddings.shape}")
    print(f"   Entity mappings: {len(entity2id):,}")
    print(f"   Relation mappings: {len(relation2id):,}")

    # Train/val split by question ID
    print("\n[4/6] Creating train/val split...")
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

    # Validation set metadata (hops are 1-indexed in raw data: 1, 2, 3)
    val_labels = [s['label'] for s in val_samples]
    val_pos_count = sum(val_labels)
    val_neg_count = len(val_labels) - val_pos_count
    val_hops = [s['hop'] for s in val_samples]
    val_hop_counts = {hop: val_hops.count(hop) for hop in [1, 2, 3]}

    print(f"\n   Validation set breakdown:")
    print(f"      Positive: {val_pos_count:,} ({100*val_pos_count/len(val_labels):.1f}%)")
    print(f"      Negative: {val_neg_count:,} ({100*val_neg_count/len(val_labels):.1f}%)")
    print(f"        Baseline (always predict negative): {100*val_neg_count/len(val_labels):.1f}% accuracy")
    print(f"        Primary metric: F1 score (balances precision & recall)")
    print(f"      Hop 1: {val_hop_counts.get(1, 0):,}")
    print(f"      Hop 2: {val_hop_counts.get(2, 0):,}")
    print(f"      Hop 3: {val_hop_counts.get(3, 0):,}")

    # Save validation set metadata
    val_metadata = {
        'num_questions': len(val_questions),
        'num_samples': len(val_samples),
        'pos_samples': val_pos_count,
        'neg_samples': val_neg_count,
        'pos_ratio': val_pos_count / len(val_labels),
        'hop_distribution': val_hop_counts,
        'question_ids': sorted(list(val_questions)),
        'split_ratio': VAL_SPLIT,
        'random_seed': 42
    }

    # Save to file
    models_dir = Config.BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    val_metadata_path = models_dir / "variant3_validation_metadata.json"
    with open(val_metadata_path, 'w') as f:
        json.dump(val_metadata, f, indent=2)
    print(f"   Validation metadata saved: {val_metadata_path}")

    # Create datasets
    print("\n[5/6] Creating PyTorch datasets...")
    train_dataset = EdgeScorerDualDataset(
        train_samples,
        text_embeddings_cache,
        entity_graph_embeddings,
        entity2id,
        relation_graph_embeddings,
        relation2id
    )
    val_dataset = EdgeScorerDualDataset(
        val_samples,
        text_embeddings_cache,
        entity_graph_embeddings,
        entity2id,
        relation_graph_embeddings,
        relation2id
    )

    # Use more workers for faster data loading with larger batch sizes
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    # Create model
    print("\n[6/6] Creating model...")
    model = EdgeScorerDual(
        text_dim=1536,
        graph_dim=256,
        hidden_dim=512,
        hop_context_dim=6,  # 6-bit one-hot encoding
        dropout=0.3
    ).to(DEVICE)

    print(f"   Parameters: {count_parameters(model):,}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(DEVICE))
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # Learning rate scheduler: Cosine annealing with warm restarts
    # T_0: epochs before first restart, T_mult: factor to increase T_0 after each restart
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

        # Scheduler (step after each epoch for CosineAnnealingWarmRestarts)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Print metrics (F1 is primary metric for imbalanced data)
        print(f"   Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f} "
              f"(P: {train_metrics['precision']:.4f}, R: {train_metrics['recall']:.4f}), "
              f"AUC: {train_metrics['auc']:.4f}")
        print(f"   Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f} "
              f"(P: {val_metrics['precision']:.4f}, R: {val_metrics['recall']:.4f}), "
              f"AUC: {val_metrics['auc']:.4f}")
        print(f"   LR: {current_lr:.6f}")

        # Per-hop metrics
        print(f"   Val per-hop metrics:")
        for hop_num in [0, 1, 2]:
            if hop_num in val_metrics['hop_metrics']:
                hm = val_metrics['hop_metrics'][hop_num]
                print(f"      Hop {hop_num + 1}: Acc={hm['accuracy']:.4f}, P={hm['precision']:.4f}, "
                      f"R={hm['recall']:.4f}, F1={hm['f1']:.4f}, AUC={hm['auc']:.4f} (n={int(hm['num_samples'])})")

        # Log to WandB
        if USE_WANDB:
            wandb_log = {
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
            }

            # Add per-hop metrics
            for hop_num in [0, 1, 2]:
                if hop_num in val_metrics['hop_metrics']:
                    hm = val_metrics['hop_metrics'][hop_num]
                    wandb_log[f'val/hop_{hop_num+1}_accuracy'] = hm['accuracy']
                    wandb_log[f'val/hop_{hop_num+1}_precision'] = hm['precision']
                    wandb_log[f'val/hop_{hop_num+1}_recall'] = hm['recall']
                    wandb_log[f'val/hop_{hop_num+1}_f1'] = hm['f1']
                    wandb_log[f'val/hop_{hop_num+1}_auc'] = hm['auc']

            wandb.log(wandb_log)

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
            checkpoint_path = Config.BASE_DIR / "models" / "variant3_edge_scorer_dual_best.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics
            }, checkpoint_path)
            print(f"    Best model saved (F1: {best_f1:.4f})")
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

    history_path = models_dir / "variant3_training_history_dual.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"Training history (JSON) saved: {history_path}")

    # Save as CSV for easy analysis
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

        # Add per-hop metrics
        for hop_num in [0, 1, 2]:
            if hop_num in entry['val_metrics']['hop_metrics']:
                hm = entry['val_metrics']['hop_metrics'][hop_num]
                row[f'val_hop{hop_num+1}_accuracy'] = hm['accuracy']
                row[f'val_hop{hop_num+1}_precision'] = hm['precision']
                row[f'val_hop{hop_num+1}_recall'] = hm['recall']
                row[f'val_hop{hop_num+1}_f1'] = hm['f1']
                row[f'val_hop{hop_num+1}_auc'] = hm['auc']
                row[f'val_hop{hop_num+1}_num_samples'] = hm['num_samples']

        csv_rows.append(row)

    df = pd.DataFrame(csv_rows)
    csv_path = models_dir / "variant3_training_history_dual.csv"
    df.to_csv(csv_path, index=False)
    print(f"Training history (CSV) saved: {csv_path}")

    # Save summary statistics for the paper
    summary_stats = {
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

    summary_path = models_dir / "variant3_training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"Training summary saved: {summary_path}")

    # Final validation analysis with best model
    print("\n[Final Validation Analysis]")
    print("Loading best model for final evaluation...")

    # Load best model
    checkpoint_path = models_dir / "variant3_edge_scorer_dual_best.pt"
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
    print(f"\n   Classification Report:")
    print(f"      Negative - P: {class_report['Negative']['precision']:.4f}, "
          f"R: {class_report['Negative']['recall']:.4f}, "
          f"F1: {class_report['Negative']['f1-score']:.4f}")
    print(f"      Positive - P: {class_report['Positive']['precision']:.4f}, "
          f"R: {class_report['Positive']['recall']:.4f}, "
          f"F1: {class_report['Positive']['f1-score']:.4f}")

    # Save detailed validation results
    val_results = {
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
        },
        'hop_metrics': final_val_metrics['hop_metrics']
    }

    val_results_path = models_dir / "variant3_validation_results.json"
    with open(val_results_path, 'w') as f:
        json.dump(val_results, f, indent=2)
    print(f"\n   Validation results saved: {val_results_path}")

    # Save predictions for further analysis
    predictions_df = pd.DataFrame({
        'label': preds_data['labels'].flatten(),
        'prediction': preds_data['preds'].flatten(),
        'probability': preds_data['probs'].flatten(),
        'hop': preds_data['hops'].flatten(),
        'correct': (preds_data['labels'].flatten() == preds_data['preds'].flatten())
    })

    predictions_path = models_dir / "variant3_validation_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"   Predictions CSV saved: {predictions_path}")
    print(f"   Total predictions: {len(predictions_df):,}")

    # Per-hop confusion matrices
    print(f"\n   Per-Hop Confusion Matrices:")
    for hop_num in [0, 1, 2]:
        hop_mask = preds_data['hops'] == hop_num
        if hop_mask.sum() > 0:
            hop_cm = confusion_matrix(
                preds_data['labels'][hop_mask],
                preds_data['preds'][hop_mask]
            )
            print(f"      Hop {hop_num + 1}: TN={hop_cm[0,0]}, FP={hop_cm[0,1]}, "
                  f"FN={hop_cm[1,0]}, TP={hop_cm[1,1]}")

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
    print("All metrics and predictions saved successfully!")
    print("=" * 80)
    print(f"\nSaved files:")
    print(f"  - {history_path}")
    print(f"  - {csv_path}")
    print(f"  - {summary_path}")
    print(f"  - {val_metadata_path}")
    print(f"  - {val_results_path}")
    print(f"  - {predictions_path}")
    print(f"  - {checkpoint_path}")


if __name__ == "__main__":
    main()
