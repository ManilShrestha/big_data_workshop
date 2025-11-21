#!/usr/bin/env python3
"""
Unified Ablation Training Script

Ablation configurations:
- abl1: Text Embeddings + Hop Count (no graph)
- abl2: Text Embeddings only (no hop, no graph)
- abl3: Graph Embeddings + Hop Count (no text)
- abl4: Graph Embeddings only (no hop, no text)

Usage:
  python ablations_train.py --ablation abl1
  python ablations_train.py --ablation abl2
  python ablations_train.py --ablation abl3
  python ablations_train.py --ablation abl4
  python ablations_train.py --ablation all  # Train all ablations sequentially
"""

import argparse
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
from dotenv import load_dotenv
import wandb

load_dotenv()

from qa_system.config import Config

# Import all model classes
from variant3.variant3_edge_scorer_abl3_text_hops import EdgeScorerTextHops  # abl1: Text + Hops
from variant3.variant3_edge_scorer_abl1_text_only import EdgeScorerTextOnly  # abl2: Text only
from variant3.variant3_edge_scorer_abl3_graph_hops import EdgeScorerGraphHops  # abl3: Graph + Hops
from variant3.variant3_edge_scorer_abl2_graph_only import EdgeScorerGraphOnly  # abl4: Graph only

# Ablation configurations
ABLATION_CONFIG = {
    'abl1': {
        'name': 'Text + Hop Count',
        'model_class': EdgeScorerTextHops,
        'needs_text': True,
        'needs_graph': False,
        'needs_hop': True,
        'wandb_name': 'ablation1-text-hops',
        'model_file': 'ablation1_text_hops_best.pt',
    },
    'abl2': {
        'name': 'Text Only',
        'model_class': EdgeScorerTextOnly,
        'needs_text': True,
        'needs_graph': False,
        'needs_hop': False,
        'wandb_name': 'ablation2-text-only',
        'model_file': 'ablation2_text_only_best.pt',
    },
    'abl3': {
        'name': 'Graph + Hop Count',
        'model_class': EdgeScorerGraphHops,
        'needs_text': False,
        'needs_graph': True,
        'needs_hop': True,
        'wandb_name': 'ablation3-graph-hops',
        'model_file': 'ablation3_graph_hops_best.pt',
    },
    'abl4': {
        'name': 'Graph Only',
        'model_class': EdgeScorerGraphOnly,
        'needs_text': False,
        'needs_graph': True,
        'needs_hop': False,
        'wandb_name': 'ablation4-graph-only',
        'model_file': 'ablation4_graph_only_best.pt',
    },
}


class AblationDataset(Dataset):
    """Unified dataset for all ablations"""

    def __init__(
        self,
        samples: list,
        config: dict,
        text_embeddings_cache: dict = None,
        entity_graph_embeddings: np.ndarray = None,
        entity2id: dict = None,
        relation_graph_embeddings: np.ndarray = None,
        relation2id: dict = None
    ):
        self.samples = samples
        self.config = config
        self.text_embeddings_cache = text_embeddings_cache
        self.entity_graph_embeddings = entity_graph_embeddings
        self.entity2id = entity2id
        self.relation_graph_embeddings = relation_graph_embeddings
        self.relation2id = relation2id

        self.zero_text = np.zeros(1536, dtype=np.float32)
        self.zero_graph = np.zeros(256, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def _get_text_emb(self, text: str):
        if self.text_embeddings_cache is None:
            return self.zero_text
        return self.text_embeddings_cache.get(text, self.zero_text)

    def _get_entity_emb(self, entity_id: str):
        if self.entity_graph_embeddings is None or self.entity2id is None:
            return self.zero_graph
        if entity_id in self.entity2id:
            idx = self.entity2id[entity_id]
            return self.entity_graph_embeddings[idx]
        return self.zero_graph

    def _get_relation_emb(self, relation: str):
        if self.relation_graph_embeddings is None or self.relation2id is None:
            return self.zero_graph
        if relation in self.relation2id:
            idx = self.relation2id[relation]
            return self.relation_graph_embeddings[idx]
        return self.zero_graph

    def __getitem__(self, idx):
        sample = self.samples[idx]

        question_text = sample['question_text']
        node_id = sample['node_id']
        relation = sample['edge_relation']
        target_id = sample['edge_target']
        hop = sample['hop'] - 1  # Convert 1-indexed to 0-indexed
        label = sample['label']

        result = {'label': torch.tensor(label, dtype=torch.float32)}

        if self.config['needs_text']:
            result['question_text_emb'] = torch.tensor(self._get_text_emb(question_text), dtype=torch.float32)
            result['node_text_emb'] = torch.tensor(self._get_text_emb(node_id), dtype=torch.float32)
            result['edge_text_emb'] = torch.tensor(self._get_text_emb(relation), dtype=torch.float32)
            result['target_text_emb'] = torch.tensor(self._get_text_emb(target_id), dtype=torch.float32)

        if self.config['needs_graph']:
            result['node_graph_emb'] = torch.tensor(self._get_entity_emb(node_id), dtype=torch.float32)
            result['edge_graph_emb'] = torch.tensor(self._get_relation_emb(relation), dtype=torch.float32)
            result['target_graph_emb'] = torch.tensor(self._get_entity_emb(target_id), dtype=torch.float32)

        if self.config['needs_hop']:
            result['hop'] = torch.tensor(hop, dtype=torch.long)

        return result


def train_epoch(model, dataloader, optimizer, criterion, device, config, log_wandb=False, epoch=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_labels = []
    all_preds = []
    all_probs = []

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(pbar):
        labels = batch['label'].to(device).unsqueeze(1)

        # Build forward args based on config
        if config['needs_text'] and config['needs_hop']:
            # abl1: Text + Hops
            logits = model(
                batch['question_text_emb'].to(device),
                batch['node_text_emb'].to(device),
                batch['edge_text_emb'].to(device),
                batch['target_text_emb'].to(device),
                batch['hop'].to(device)
            )
        elif config['needs_text'] and not config['needs_hop']:
            # abl2: Text only
            logits = model(
                batch['question_text_emb'].to(device),
                batch['node_text_emb'].to(device),
                batch['edge_text_emb'].to(device),
                batch['target_text_emb'].to(device)
            )
        elif config['needs_graph'] and config['needs_hop']:
            # abl3: Graph + Hops
            logits = model(
                batch['node_graph_emb'].to(device),
                batch['edge_graph_emb'].to(device),
                batch['target_graph_emb'].to(device),
                batch['hop'].to(device)
            )
        elif config['needs_graph'] and not config['needs_hop']:
            # abl4: Graph only
            logits = model(
                batch['node_graph_emb'].to(device),
                batch['edge_graph_emb'].to(device),
                batch['target_graph_emb'].to(device)
            )

        optimizer.zero_grad()
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        if log_wandb and batch_idx % 100 == 0:
            wandb.log({
                'train/batch_loss': loss.item(),
                'train/batch': epoch * len(dataloader) + batch_idx
            })

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    return {'loss': avg_loss, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}


def evaluate(model, dataloader, criterion, device, config, return_predictions=False):
    """Evaluate on validation set"""
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in pbar:
            labels = batch['label'].to(device).unsqueeze(1)

            if config['needs_text'] and config['needs_hop']:
                logits = model(
                    batch['question_text_emb'].to(device),
                    batch['node_text_emb'].to(device),
                    batch['edge_text_emb'].to(device),
                    batch['target_text_emb'].to(device),
                    batch['hop'].to(device)
                )
            elif config['needs_text'] and not config['needs_hop']:
                logits = model(
                    batch['question_text_emb'].to(device),
                    batch['node_text_emb'].to(device),
                    batch['edge_text_emb'].to(device),
                    batch['target_text_emb'].to(device)
                )
            elif config['needs_graph'] and config['needs_hop']:
                logits = model(
                    batch['node_graph_emb'].to(device),
                    batch['edge_graph_emb'].to(device),
                    batch['target_graph_emb'].to(device),
                    batch['hop'].to(device)
                )
            elif config['needs_graph'] and not config['needs_hop']:
                logits = model(
                    batch['node_graph_emb'].to(device),
                    batch['edge_graph_emb'].to(device),
                    batch['target_graph_emb'].to(device)
                )

            loss = criterion(logits, labels)
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    results = {'loss': avg_loss, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}

    if return_predictions:
        results['predictions'] = {
            'labels': np.array(all_labels),
            'preds': np.array(all_preds),
            'probs': np.array(all_probs)
        }

    return results


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_ablation(ablation_name: str, use_wandb: bool = True, num_epochs: int = 50):
    """Train a single ablation"""
    config = ABLATION_CONFIG[ablation_name]

    print("=" * 80)
    print(f"Training {ablation_name.upper()}: {config['name']}")
    print("=" * 80)

    # Hyperparameters
    BATCH_SIZE = 2048
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = num_epochs
    PATIENCE = 10
    VAL_SPLIT = 0.2
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n[Configuration]")
    print(f"   Device: {DEVICE}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Max epochs: {NUM_EPOCHS}")
    print(f"   Early stopping patience: {PATIENCE}")

    # Initialize WandB
    if use_wandb:
        wandb.init(
            project="big-data-qa-ablations",
            name=config['wandb_name'],
            config={
                "ablation": ablation_name,
                "ablation_name": config['name'],
                "needs_text": config['needs_text'],
                "needs_graph": config['needs_graph'],
                "needs_hop": config['needs_hop'],
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "max_epochs": NUM_EPOCHS,
                "patience": PATIENCE,
            }
        )

    # Load training data
    print("\n[1/5] Loading training data...")
    training_data_path = Config.BASE_DIR / "data/variant3/variant3_training_data.pkl"
    with open(training_data_path, 'rb') as f:
        all_samples = pickle.load(f)
    print(f"   Total samples: {len(all_samples):,}")

    # Load embeddings based on config
    text_embeddings_cache = None
    entity_graph_embeddings = None
    entity2id = None
    relation_graph_embeddings = None
    relation2id = None

    if config['needs_text']:
        print("\n[2/5] Loading text embeddings...")
        with open(Config.EMBEDDINGS_DIR / "variant3_text_embeddings_cache.pkl", 'rb') as f:
            text_embeddings_cache = pickle.load(f)
        print(f"   Text embeddings: {len(text_embeddings_cache):,} entries")

    if config['needs_graph']:
        print("\n[2/5] Loading graph embeddings (TransE)...")
        entity_graph_embeddings = np.load(Config.TRANSE_ENTITY_PATH)
        relation_graph_embeddings = np.load(Config.TRANSE_RELATION_PATH)

        with open(Config.EMBEDDINGS_DIR / "node2id.json", 'r') as f:
            entity2id = json.load(f)

        with open(Config.TRANSE_METADATA_PATH, 'r') as f:
            transe_metadata = json.load(f)

        relation2id_path = Config.EMBEDDINGS_DIR / "relation2id.json"
        if relation2id_path.exists():
            with open(relation2id_path, 'r') as f:
                relation2id = json.load(f)
        else:
            relation2id = transe_metadata.get('relation_to_id', transe_metadata.get('relation2id', {}))

        print(f"   Entity embeddings: {entity_graph_embeddings.shape}")
        print(f"   Relation embeddings: {relation_graph_embeddings.shape}")

    # Train/val split
    print("\n[3/5] Creating train/val split...")
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
    print(f"   Pos weight: {pos_weight:.2f}")

    # Create datasets
    print("\n[4/5] Creating datasets...")
    train_dataset = AblationDataset(
        train_samples, config,
        text_embeddings_cache, entity_graph_embeddings, entity2id,
        relation_graph_embeddings, relation2id
    )
    val_dataset = AblationDataset(
        val_samples, config,
        text_embeddings_cache, entity_graph_embeddings, entity2id,
        relation_graph_embeddings, relation2id
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    # Create model
    print("\n[5/5] Creating model...")
    ModelClass = config['model_class']

    if config['needs_text'] and config['needs_hop']:
        model = ModelClass(text_dim=1536, hidden_dim=512, max_hops=3, dropout=0.3)
    elif config['needs_text'] and not config['needs_hop']:
        model = ModelClass(text_dim=1536, hidden_dim=512, dropout=0.3)
    elif config['needs_graph'] and config['needs_hop']:
        model = ModelClass(graph_dim=256, hidden_dim=512, max_hops=3, dropout=0.3)
    elif config['needs_graph'] and not config['needs_hop']:
        model = ModelClass(graph_dim=256, hidden_dim=512, dropout=0.3)

    model = model.to(DEVICE)
    print(f"   Parameters: {count_parameters(model):,}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(DEVICE))
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # Training loop
    print("\n[Training...]")
    best_f1 = 0
    patience_counter = 0
    training_history = []

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, DEVICE, config,
                                    log_wandb=use_wandb, epoch=epoch)
        val_metrics = evaluate(model, val_loader, criterion, DEVICE, config)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"   Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}")
        print(f"   Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")

        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/f1': train_metrics['f1'],
                'train/auc': train_metrics['auc'],
                'val/loss': val_metrics['loss'],
                'val/f1': val_metrics['f1'],
                'val/auc': val_metrics['auc'],
                'learning_rate': current_lr,
            })

        training_history.append({
            'epoch': epoch + 1,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'learning_rate': current_lr
        })

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0

            checkpoint_path = Config.BASE_DIR / "models" / config['model_file']
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics,
                'config': config
            }, checkpoint_path)
            print(f"   Best model saved (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n   Early stopping triggered (patience={PATIENCE})")
                break

    # Save training history
    models_dir = Config.BASE_DIR / "models"
    history_path = models_dir / f"{ablation_name}_training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)

    # Save CSV
    csv_rows = []
    for entry in training_history:
        row = {
            'epoch': entry['epoch'],
            'learning_rate': entry['learning_rate'],
            'train_loss': entry['train_metrics']['loss'],
            'train_f1': entry['train_metrics']['f1'],
            'train_auc': entry['train_metrics']['auc'],
            'val_loss': entry['val_metrics']['loss'],
            'val_f1': entry['val_metrics']['f1'],
            'val_auc': entry['val_metrics']['auc'],
        }
        csv_rows.append(row)

    df = pd.DataFrame(csv_rows)
    csv_path = models_dir / f"{ablation_name}_training_history.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print(f"{ablation_name.upper()} Training Complete!")
    print(f"Best validation F1: {best_f1:.4f}")
    print("=" * 80)

    if use_wandb:
        wandb.finish()

    return best_f1


def main():
    parser = argparse.ArgumentParser(description='Train ablation models')
    parser.add_argument('--ablation', type=str, required=True,
                       choices=['abl1', 'abl2', 'abl3', 'abl4', 'all'],
                       help='Which ablation to train')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB logging')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')

    args = parser.parse_args()

    use_wandb = not args.no_wandb

    if args.ablation == 'all':
        results = {}
        for abl in ['abl1', 'abl2', 'abl3', 'abl4']:
            results[abl] = train_ablation(abl, use_wandb=use_wandb, num_epochs=args.epochs)

        print("\n" + "=" * 80)
        print("ALL ABLATIONS COMPLETE")
        print("=" * 80)
        for abl, f1 in results.items():
            config = ABLATION_CONFIG[abl]
            print(f"  {abl}: {config['name']} -> F1: {f1:.4f}")
    else:
        train_ablation(args.ablation, use_wandb=use_wandb, num_epochs=args.epochs)


if __name__ == "__main__":
    main()
