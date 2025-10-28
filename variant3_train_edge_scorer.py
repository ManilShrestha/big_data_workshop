"""
Train EdgeScorer neural network for Variant 3

Training procedure:
1. Load training data (question, node, edge, hop, label)
2. Load cached embeddings (question, node, edge)
3. Create PyTorch dataset with train/val split
4. Train with BCEWithLogitsLoss + pos_weight for class imbalance
5. Evaluate on validation set
6. Save best model checkpoint

Key features:
- Class balancing via pos_weight
- Early stopping based on validation F1
- Learning rate scheduling
- Per-hop accuracy tracking
"""

import pickle
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time

from qa_system.config import Config
from variant3_edge_scorer import EdgeScorer, count_parameters


class EdgeScorerDataset(Dataset):
    """PyTorch dataset for EdgeScorer training"""

    def __init__(
        self,
        samples: list,
        question_embeddings: dict,
        node_embeddings: np.ndarray,
        node2id: dict,
        edge_embeddings: np.ndarray,
        relation2id: dict
    ):
        """
        Args:
            samples: List of training samples from variant3_create_training_data.py
            question_embeddings: Dict mapping question_text -> embedding
            node_embeddings: TransE entity embeddings [num_entities, 256]
            node2id: Dict mapping entity name -> entity ID
            edge_embeddings: TransE relation embeddings [num_relations, 256]
            relation2id: Dict mapping relation name -> relation ID
        """
        self.samples = samples
        self.question_embeddings = question_embeddings
        self.node_embeddings = node_embeddings
        self.node2id = node2id
        self.edge_embeddings = edge_embeddings
        self.relation2id = relation2id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Get embeddings
        question_text = sample['question_text']
        node_id = sample['node_id']
        relation = sample['edge_relation']
        hop = sample['hop']
        label = sample['label']

        # Question embedding (OpenAI)
        q_emb = self.question_embeddings[question_text]

        # Node embedding (TransE)
        if node_id in self.node2id:
            node_idx = self.node2id[node_id]
            n_emb = self.node_embeddings[node_idx]
        else:
            n_emb = np.zeros(256, dtype=np.float32)  # Unknown node

        # Edge embedding (TransE)
        if relation in self.relation2id:
            rel_idx = self.relation2id[relation]
            e_emb = self.edge_embeddings[rel_idx]
        else:
            e_emb = np.zeros(256, dtype=np.float32)  # Unknown relation

        return {
            'question_emb': torch.tensor(q_emb, dtype=torch.float32),
            'node_emb': torch.tensor(n_emb, dtype=torch.float32),
            'edge_emb': torch.tensor(e_emb, dtype=torch.float32),
            'hop': torch.tensor(hop, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float32)
        }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_labels = []
    all_preds = []

    for batch in tqdm(dataloader, desc="Training", leave=False):
        # Move to device
        question_emb = batch['question_emb'].to(device)
        node_emb = batch['node_emb'].to(device)
        edge_emb = batch['edge_emb'].to(device)
        hop = batch['hop'].to(device)
        labels = batch['label'].to(device).unsqueeze(1)

        # Forward pass
        optimizer.zero_grad()
        logits = model(question_emb, node_emb, edge_emb, hop)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Track predictions
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation set"""
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    all_probs = []
    all_hops = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            question_emb = batch['question_emb'].to(device)
            node_emb = batch['node_emb'].to(device)
            edge_emb = batch['edge_emb'].to(device)
            hop = batch['hop'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)

            logits = model(question_emb, node_emb, edge_emb, hop)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_hops.extend(hop.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    # Overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # Per-hop metrics
    hop_metrics = {}
    for hop_num in [0, 1, 2]:
        hop_mask = np.array(all_hops) == hop_num
        if hop_mask.sum() > 0:
            hop_labels = np.array(all_labels)[hop_mask]
            hop_preds = np.array(all_preds)[hop_mask]
            hop_acc = accuracy_score(hop_labels, hop_preds)
            hop_recall = recall_score(hop_labels, hop_preds, zero_division=0)
            hop_metrics[hop_num] = {'accuracy': hop_acc, 'recall': hop_recall}

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'hop_metrics': hop_metrics
    }


def main():
    print("=" * 80)
    print("Variant 3: Train EdgeScorer Neural Network")
    print("=" * 80)

    # Hyperparameters
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50
    PATIENCE = 5  # Early stopping patience
    VAL_SPLIT = 0.2
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n[Configuration]")
    print(f"   Device: {DEVICE}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Max epochs: {NUM_EPOCHS}")
    print(f"   Early stopping patience: {PATIENCE}")

    # Load training data
    print("\n[1/6] Loading training data...")
    training_data_path = Config.BASE_DIR / "data" / "variant3_training_data.pkl"
    with open(training_data_path, 'rb') as f:
        all_samples = pickle.load(f)
    print(f"   Total samples: {len(all_samples)}")

    # Load cached question embeddings
    print("\n[2/6] Loading cached embeddings...")
    question_emb_path = Config.EMBEDDINGS_DIR / "variant3_question_embeddings_train.pkl"
    with open(question_emb_path, 'rb') as f:
        question_embeddings = pickle.load(f)
    print(f"   Question embeddings: {len(question_embeddings)}")

    # Load TransE embeddings
    node_embeddings = np.load(Config.TRANSE_ENTITY_PATH)
    edge_embeddings = np.load(Config.TRANSE_RELATION_PATH)
    print(f"   Node embeddings: {node_embeddings.shape}")
    print(f"   Edge embeddings: {edge_embeddings.shape}")

    # Load metadata
    with open(Config.TRANSE_METADATA_PATH, 'r') as f:
        transe_metadata = json.load(f)
    node2id = transe_metadata['entity2id']
    relation2id = transe_metadata['relation2id']

    # Split by question ID to avoid leakage
    print("\n[3/6] Creating train/val split...")
    unique_questions = list(set(s['question_id'] for s in all_samples))
    np.random.seed(42)
    np.random.shuffle(unique_questions)

    split_idx = int(len(unique_questions) * (1 - VAL_SPLIT))
    train_questions = set(unique_questions[:split_idx])
    val_questions = set(unique_questions[split_idx:])

    train_samples = [s for s in all_samples if s['question_id'] in train_questions]
    val_samples = [s for s in all_samples if s['question_id'] in val_questions]

    print(f"   Train samples: {len(train_samples)}")
    print(f"   Val samples: {len(val_samples)}")

    # Compute class weights
    train_labels = [s['label'] for s in train_samples]
    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    pos_weight = neg_count / pos_count
    print(f"   Positive samples: {pos_count}")
    print(f"   Negative samples: {neg_count}")
    print(f"   Pos weight: {pos_weight:.2f}")

    # Create datasets
    print("\n[4/6] Creating PyTorch datasets...")
    train_dataset = EdgeScorerDataset(
        train_samples, question_embeddings,
        node_embeddings, node2id,
        edge_embeddings, relation2id
    )
    val_dataset = EdgeScorerDataset(
        val_samples, question_embeddings,
        node_embeddings, node2id,
        edge_embeddings, relation2id
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Create model
    print("\n[5/6] Creating model...")
    model = EdgeScorer(
        question_dim=1536,
        node_dim=256,
        edge_dim=256,
        hidden_dim=512,
        max_hops=3,
        dropout=0.3
    ).to(DEVICE)
    print(f"   Parameters: {count_parameters(model):,}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(DEVICE))
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    # Training loop
    print("\n[6/6] Training...")
    best_f1 = 0
    patience_counter = 0
    training_history = []

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)

        # Scheduler step
        scheduler.step(val_metrics['f1'])

        # Print metrics
        print(f"   Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"   Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"P: {val_metrics['precision']:.4f}, R: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")

        # Per-hop metrics
        print(f"   Val per-hop accuracy:")
        for hop_num in [0, 1, 2]:
            if hop_num in val_metrics['hop_metrics']:
                hop_acc = val_metrics['hop_metrics'][hop_num]['accuracy']
                hop_recall = val_metrics['hop_metrics'][hop_num]['recall']
                print(f"      Hop {hop_num + 1}: Acc={hop_acc:.4f}, Recall={hop_recall:.4f}")

        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_metrics': val_metrics
        })

        # Early stopping
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0

            # Save best model
            checkpoint_path = Config.BASE_DIR / "models" / "variant3_edge_scorer_best.pt"
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

    history_path = Config.BASE_DIR / "models" / "variant3_training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()
