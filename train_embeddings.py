"""
Modular training script for graph embeddings with argparse.

Usage:
    python train_embeddings.py --method fastrp
    python train_embeddings.py --method node2vec --walks 200
    python train_embeddings.py --method transe --epochs 100
    python train_embeddings.py --method all
"""

import pickle
import numpy as np
import networkx as nx
from pathlib import Path
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Tuple, List
import logging

# Check available libraries
try:
    from node2vec import Node2Vec
    NODE2VEC_AVAILABLE = True
except ImportError:
    NODE2VEC_AVAILABLE = False

try:
    import torch
    from pykeen.triples import TriplesFactory
    from pykeen.pipeline import pipeline
    PYKEEN_AVAILABLE = True
except ImportError:
    PYKEEN_AVAILABLE = False


class TimingLogger:
    """Helper class for logging with timestamps."""

    def __init__(self, log_file: str = None):
        self.log_file = log_file
        self.start_time = time.time()

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file) if log_file else logging.NullHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        elapsed = time.time() - self.start_time
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamp = f"[{current_time}] [{elapsed:7.2f}s]"
        full_message = f"{timestamp} {message}"

        if level == "INFO":
            self.logger.info(full_message)
        elif level == "ERROR":
            self.logger.error(full_message)
        elif level == "WARNING":
            self.logger.warning(full_message)


class EmbeddingTrainer:
    """Train embeddings for MetaQA knowledge graph."""

    def __init__(self, graph_path: str = "data/metaqa/graph.pkl", logger: TimingLogger = None):
        """Load the graph."""
        self.logger = logger or TimingLogger()

        self.logger.log(f"📂 Loading graph from {graph_path}...")
        with open(graph_path, 'rb') as f:
            self.G = pickle.load(f)

        self.num_nodes = self.G.number_of_nodes()
        self.num_edges = self.G.number_of_edges()
        self.logger.log(f"✓ Loaded graph: {self.num_nodes:,} nodes, {self.num_edges:,} edges")

        # Create node to ID mapping
        self.node_list = list(self.G.nodes())
        self.node2id = {node: idx for idx, node in enumerate(self.node_list)}
        self.id2node = {idx: node for node, idx in self.node2id.items()}

    def train_fastrp(self, embedding_dim: int = 128,
                     iterations: int = 5,
                     normalization_strength: float = 0.5,
                     checkpoints: List[int] = None) -> List[Tuple[np.ndarray, Dict]]:
        """
        Generate FastRP embeddings with optional checkpoints.

        Args:
            embedding_dim: Dimension of embeddings
            iterations: Maximum number of propagation iterations
            normalization_strength: L2 normalization strength (0-1)
            checkpoints: List of iteration counts to save (if None, only saves final)
        """
        self.logger.log("="*60)
        self.logger.log("🚀 Training FastRP Embeddings")
        self.logger.log("="*60)

        if checkpoints is None:
            checkpoints = [iterations]

        self.logger.log(f"   Parameters: dim={embedding_dim}, iterations={iterations}")
        self.logger.log(f"   normalization_strength={normalization_strength}")
        if len(checkpoints) > 1:
            self.logger.log(f"   Checkpoints at: {checkpoints} iterations")
        self.logger.log("")

        # Initialize random embeddings
        np.random.seed(42)
        embeddings = np.random.randn(self.num_nodes, embedding_dim).astype(np.float32)

        # Normalize initial embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

        # Create adjacency structure
        self.logger.log("   Building adjacency structure...")
        adj = {i: [] for i in range(self.num_nodes)}
        for u, v in self.G.edges():
            u_id = self.node2id[u]
            v_id = self.node2id[v]
            adj[u_id].append(v_id)
            adj[v_id].append(u_id)

        results = []
        start_time = time.time()

        # Iterative propagation with checkpoints
        self.logger.log(f"   Propagating for up to {iterations} iterations...")
        for iteration in range(iterations):
            new_embeddings = embeddings.copy()

            for node_id in range(self.num_nodes):
                neighbors = adj[node_id]
                if neighbors:
                    neighbor_emb = embeddings[neighbors].mean(axis=0)
                    new_embeddings[node_id] = (
                        (1 - normalization_strength) * embeddings[node_id] +
                        normalization_strength * neighbor_emb
                    )

            embeddings = new_embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

            current_iteration = iteration + 1

            # Check if this is a checkpoint
            if current_iteration in checkpoints:
                checkpoint_time = time.time() - start_time

                if len(checkpoints) > 1:
                    self.logger.log(f"      📍 Checkpoint: {current_iteration} iterations")
                else:
                    self.logger.log(f"      Iteration {current_iteration}/{iterations} complete")

                self.logger.log(f"      ✓ Complete in {checkpoint_time:.2f}s")
                self.logger.log(f"      Shape: {embeddings.shape}")
                self.logger.log(f"      Mean norm: {np.linalg.norm(embeddings, axis=1).mean():.4f}")
                self.logger.log("")

                metadata = {
                    "method": "FastRP",
                    "embedding_dim": embedding_dim,
                    "iterations": current_iteration,
                    "normalization_strength": normalization_strength,
                    "num_nodes": self.num_nodes,
                    "training_time_seconds": checkpoint_time,
                    "checkpoint": len(checkpoints) > 1,
                    "timestamp": datetime.now().isoformat()
                }

                # Save a copy of embeddings at this checkpoint
                results.append((embeddings.copy(), metadata))
            elif (iteration + 1) % 2 == 0:
                self.logger.log(f"      Iteration {current_iteration}/{iterations} complete")

        elapsed = time.time() - start_time
        self.logger.log(f"✓ FastRP training complete in {elapsed:.2f}s")

        return results

    def train_node2vec(self,
                      embedding_dim: int = 128,
                      walk_length: int = 30,
                      num_walks: int = 200,
                      p: float = 1.0,
                      q: float = 1.0,
                      workers: int = 8,
                      epochs: int = 10,
                      checkpoints: List[int] = None) -> List[Tuple[np.ndarray, Dict]]:
        """
        Generate Node2Vec embeddings with optional checkpoints.

        Args:
            embedding_dim: Dimension of embeddings
            walk_length: Length of each random walk
            num_walks: Number of walks per node
            p: Return parameter
            q: In-out parameter
            workers: Number of parallel workers
            epochs: Number of Word2Vec training epochs
            checkpoints: List of walk counts to save (if None, only saves final)
        """
        if not NODE2VEC_AVAILABLE:
            raise ImportError("node2vec not installed. Run: pip install node2vec")

        self.logger.log("="*60)
        self.logger.log("🚀 Training Node2Vec Embeddings")
        self.logger.log("="*60)

        G_undirected = self.G.to_undirected()
        results = []

        if checkpoints is None:
            checkpoints = [num_walks]

        self.logger.log(f"   Parameters: dim={embedding_dim}, walks={num_walks}, length={walk_length}")
        self.logger.log(f"   p={p}, q={q}, workers={workers}, epochs={epochs}")
        if len(checkpoints) > 1:
            self.logger.log(f"   Checkpoints at: {checkpoints} walks")
        self.logger.log("")

        for walks in sorted(checkpoints):
            if walks > num_walks:
                continue

            checkpoint_start = time.time()

            if len(checkpoints) > 1:
                self.logger.log(f"   📍 Checkpoint: {walks} walks")

            # Initialize node2vec
            node2vec = Node2Vec(
                G_undirected,
                dimensions=embedding_dim,
                walk_length=walk_length,
                num_walks=walks,
                p=p,
                q=q,
                workers=workers,
                seed=42,
                quiet=False
            )

            # Train model
            self.logger.log(f"      Training Word2Vec model ({epochs} epochs)...")
            model = node2vec.fit(
                window=10,
                min_count=1,
                batch_words=4,
                epochs=epochs
            )

            # Extract embeddings
            embeddings = np.zeros((self.num_nodes, embedding_dim), dtype=np.float32)
            for node, idx in self.node2id.items():
                embeddings[idx] = model.wv[node]

            elapsed = time.time() - checkpoint_start
            self.logger.log(f"      ✓ Complete in {elapsed:.2f}s ({elapsed/60:.1f} min)")
            self.logger.log(f"      Shape: {embeddings.shape}")
            self.logger.log(f"      Mean norm: {np.linalg.norm(embeddings, axis=1).mean():.4f}")
            self.logger.log("")

            metadata = {
                "method": "Node2Vec",
                "embedding_dim": embedding_dim,
                "walk_length": walk_length,
                "num_walks": walks,
                "p": p,
                "q": q,
                "workers": workers,
                "w2v_epochs": epochs,
                "num_nodes": self.num_nodes,
                "training_time_seconds": elapsed,
                "checkpoint": len(checkpoints) > 1,
                "timestamp": datetime.now().isoformat()
            }

            results.append((embeddings, metadata))

        self.logger.log(f"✓ Node2Vec training complete!")
        return results

    def train_transe(self,
                    embedding_dim: int = 128,
                    num_epochs: int = 100,
                    batch_size: int = 2048,
                    learning_rate: float = 0.01,
                    patience: int = 10,
                    checkpoints: List[int] = None):
        """
        Generate TransE embeddings using GPU with optional checkpoints.

        Args:
            embedding_dim: Dimension of embeddings
            num_epochs: Maximum training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            patience: Early stopping patience (stop if no improvement for N epochs)
            checkpoints: List of epoch numbers to save checkpoints (e.g., [50, 100, 200])
                        If None, only saves final model

        Yields:
            Tuple of (embeddings, metadata, relation_embeddings, triples_factory) for each checkpoint
        """
        if not PYKEEN_AVAILABLE:
            raise ImportError("pykeen not installed. Run: pip install pykeen torch")

        self.logger.log("="*60)
        self.logger.log("🚀 Training TransE Embeddings (GPU Accelerated)")
        self.logger.log("="*60)
        start_time = time.time()

        # Convert graph to triples
        self.logger.log("   Converting graph to triples...")
        triples = []
        for u, v, data in self.G.edges(data=True):
            relation = data.get('relation', 'related_to')
            triples.append([str(u), str(relation), str(v)])

        self.logger.log(f"   Total triples: {len(triples):,}")

        # Create triples factory
        triples_array = np.array(triples)
        tf = TriplesFactory.from_labeled_triples(triples_array)

        # Split for training (use 90% train, 10% validation)
        train_tf, test_tf = tf.split([0.9, 0.1], random_state=42)

        self.logger.log(f"   Entities: {tf.num_entities:,}")
        self.logger.log(f"   Relations: {tf.num_relations:,}")
        self.logger.log(f"   Training triples: {train_tf.num_triples:,}")
        self.logger.log(f"   Validation triples: {test_tf.num_triples:,}")

        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.log(f"   🚀 Device: {device.type.upper()}")

        if device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.log(f"   🎮 GPU: {gpu_name} ({gpu_memory:.1f} GB)")

        # Determine checkpoints
        if checkpoints is None:
            checkpoints = [num_epochs]  # Only final model
        else:
            checkpoints = sorted([cp for cp in checkpoints if cp <= num_epochs])
            if not checkpoints or checkpoints[-1] != num_epochs:
                checkpoints.append(num_epochs)  # Always include final

        self.logger.log(f"   Training for {num_epochs} epochs, batch_size={batch_size}")
        self.logger.log(f"   Checkpoints at epochs: {checkpoints}")
        self.logger.log("")

        # If checkpoints, need to train incrementally
        if len(checkpoints) > 1 or checkpoints[0] < num_epochs:
            self.logger.log("   Training with checkpoints (incremental training)...")
            results = []
            trained_epochs = 0

            # Train incrementally to each checkpoint
            for i, target_epoch in enumerate(checkpoints):
                epochs_to_train = target_epoch - trained_epochs

                if epochs_to_train <= 0:
                    continue

                self.logger.log(f"\n   📊 Training to epoch {target_epoch} ({epochs_to_train} epochs)...")

                if i == 0:
                    # First training session
                    result = pipeline(
                        training=train_tf,
                        testing=test_tf,
                        model='TransE',
                        model_kwargs=dict(embedding_dim=embedding_dim),
                        optimizer='Adam',
                        optimizer_kwargs=dict(lr=learning_rate),
                        training_kwargs=dict(
                            num_epochs=epochs_to_train,
                            batch_size=batch_size,
                        ),
                        training_loop='sLCWA',
                        negative_sampler='basic',
                        random_seed=42,
                        device=device.type,
                    )
                    model = result.model
                else:
                    # Continue training existing model manually
                    self.logger.log(f"      Continuing from epoch {trained_epochs}...")

                    from pykeen.training import SLCWATrainingLoop
                    from pykeen.sampling import BasicNegativeSampler
                    from pykeen.evaluation import RankBasedEvaluator

                    # Create training loop with existing model
                    training_loop = SLCWATrainingLoop(
                        model=model,
                        triples_factory=train_tf,
                        optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate),
                    )

                    # Train additional epochs
                    training_loop.train(
                        triples_factory=train_tf,
                        num_epochs=epochs_to_train,
                        batch_size=batch_size,
                    )

                    # Evaluate
                    evaluator = RankBasedEvaluator()
                    result_obj = evaluator.evaluate(
                        model=model,
                        mapped_triples=test_tf.mapped_triples,
                        batch_size=batch_size,
                        additional_filter_triples=[train_tf.mapped_triples],
                    )

                    # Create result object
                    result = type('Result', (), {'model': model, 'metric_results': result_obj})()

                trained_epochs = target_epoch

                # Extract embeddings at this checkpoint (use result.model to ensure we get the updated model)
                entity_embeddings = result.model.entity_representations[0](indices=None).detach().cpu().numpy()
                relation_embeddings = result.model.relation_representations[0](indices=None).detach().cpu().numpy()

                # Map back to our node order
                embeddings = np.zeros((self.num_nodes, embedding_dim), dtype=np.float32)
                entity_to_id = train_tf.entity_to_id

                for node, idx in self.node2id.items():
                    node_str = str(node)
                    if node_str in entity_to_id:
                        transe_id = entity_to_id[node_str]
                        embeddings[idx] = entity_embeddings[transe_id]

                checkpoint_elapsed = time.time() - start_time
                self.logger.log(f"   ✓ Checkpoint {target_epoch} epochs: {checkpoint_elapsed:.2f}s")

                # Log metrics for this checkpoint
                if hasattr(result, 'metric_results') and result.metric_results:
                    metrics_dict = result.metric_results.to_dict()
                    # Access nested dictionary correctly
                    hits_10 = metrics_dict.get('both', {}).get('realistic', {}).get('hits_at_10', 0)
                    self.logger.log(f"      Hits@10: {hits_10:.4f}")

                metadata = {
                    "method": "TransE",
                    "embedding_dim": embedding_dim,
                    "num_epochs": target_epoch,
                    "total_planned_epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "num_nodes": self.num_nodes,
                    "num_relations": tf.num_relations,
                    "training_time_seconds": checkpoint_elapsed,
                    "device": device.type,
                    "timestamp": datetime.now().isoformat(),
                    "relation_to_id": train_tf.relation_to_id,
                    "checkpoint": True
                }

                # Add metrics to metadata
                if hasattr(result, 'metric_results') and result.metric_results:
                    metadata['evaluation_metrics'] = result.metric_results.to_dict()

                results.append((embeddings, metadata, relation_embeddings, train_tf))

            elapsed = time.time() - start_time
            self.logger.log("")
            self.logger.log(f"✓ TransE complete in {elapsed:.2f}s ({elapsed/60:.1f} min)")
            self.logger.log(f"   Total checkpoints saved: {len(results)}")

            return results

        else:
            # Single training run (no checkpoints)
            self.logger.log("   Training TransE model...")

            result = pipeline(
                training=train_tf,
                testing=test_tf,
                model='TransE',
                model_kwargs=dict(embedding_dim=embedding_dim),
                optimizer='Adam',
                optimizer_kwargs=dict(lr=learning_rate),
                training_kwargs=dict(
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                ),
                training_loop='sLCWA',
                negative_sampler='basic',
                random_seed=42,
                device=device.type,
            )

            self.logger.log(f"   ℹ️  Training completed {num_epochs} epochs")

            # Log final metrics
            self.logger.log("")
            self.logger.log("   Final evaluation metrics:")
            if result.metric_results:
                metrics_dict = result.metric_results.to_dict()
                for metric_name, metric_value in sorted(metrics_dict.items()):
                    if any(x in metric_name for x in ['hits_at', 'mean_rank', 'mean_reciprocal_rank']):
                        self.logger.log(f"      {metric_name}: {metric_value:.4f}")

            # Extract entity embeddings
            entity_embeddings = result.model.entity_representations[0](
                indices=None
            ).detach().cpu().numpy()

            # Extract relation embeddings
            relation_embeddings = result.model.relation_representations[0](
                indices=None
            ).detach().cpu().numpy()

            # Map back to our node order
            embeddings = np.zeros((self.num_nodes, embedding_dim), dtype=np.float32)
            entity_to_id = train_tf.entity_to_id

            for node, idx in self.node2id.items():
                node_str = str(node)
                if node_str in entity_to_id:
                    transe_id = entity_to_id[node_str]
                    embeddings[idx] = entity_embeddings[transe_id]

            elapsed = time.time() - start_time
            self.logger.log("")
            self.logger.log(f"✓ TransE complete in {elapsed:.2f}s ({elapsed/60:.1f} min)")
            self.logger.log(f"   Shape: {embeddings.shape}")
            self.logger.log(f"   Mean norm: {np.linalg.norm(embeddings, axis=1).mean():.4f}")

            metadata = {
                "method": "TransE",
                "embedding_dim": embedding_dim,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_nodes": self.num_nodes,
                "num_relations": tf.num_relations,
                "training_time_seconds": elapsed,
                "device": device.type,
                "timestamp": datetime.now().isoformat(),
                "relation_to_id": train_tf.relation_to_id,
                "checkpoint": False
            }

            # Add metrics to metadata
            if result.metric_results:
                metadata['evaluation_metrics'] = result.metric_results.to_dict()

            return [(embeddings, metadata, relation_embeddings, train_tf)]

    def save_embeddings(self, embeddings: np.ndarray, metadata: Dict,
                       output_dir: str = "embeddings", relation_embeddings: np.ndarray = None):
        """Save embeddings and metadata."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        method_name = metadata['method'].lower()

        # Add suffix for checkpoints
        if metadata.get('checkpoint', False):
            if 'num_walks' in metadata:
                suffix = f"_walks{metadata['num_walks']}"
            elif 'iterations' in metadata:
                suffix = f"_iter{metadata['iterations']}"
            elif 'num_epochs' in metadata and method_name == 'transe':
                suffix = f"_epochs{metadata['num_epochs']}"
            else:
                suffix = ""
        else:
            suffix = ""

        # Save embeddings
        emb_file = output_path / f"{method_name}_embeddings{suffix}.npy"
        np.save(emb_file, embeddings)
        self.logger.log(f"   💾 Saved embeddings: {emb_file}")

        # Save relation embeddings (for TransE)
        if relation_embeddings is not None:
            rel_file = output_path / f"{method_name}_relation_embeddings{suffix}.npy"
            np.save(rel_file, relation_embeddings)
            self.logger.log(f"   💾 Saved relation embeddings: {rel_file}")

        # Save node mapping (only once)
        node2id_file = output_path / "node2id.json"
        if not node2id_file.exists():
            with open(node2id_file, 'w') as f:
                json.dump(self.node2id, f, indent=2)
            self.logger.log(f"   💾 Saved node mapping: {node2id_file}")

        # Save metadata
        meta_file = output_path / f"{method_name}_metadata{suffix}.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.log(f"   💾 Saved metadata: {meta_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Train graph embeddings for MetaQA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train FastRP with default 5 iterations
  python train_embeddings.py --method fastrp

  # Train FastRP with ablation study (multiple iteration checkpoints)
  python train_embeddings.py --method fastrp --iterations 5 --iteration-checkpoints 1 2 3 4 5

  # Train Node2Vec with 200 walks and 20 epochs
  python train_embeddings.py --method node2vec --walks 200 --epochs 20

  # Train Node2Vec with checkpoints
  python train_embeddings.py --method node2vec --walks 200 --checkpoints 10 50 100 200 --epochs 20

  # Train TransE with 100 epochs and early stopping
  python train_embeddings.py --method transe --epochs 100 --early-stopping 10

  # Train all methods
  python train_embeddings.py --method all
        """
    )

    parser.add_argument(
        '--method',
        type=str,
        required=True,
        choices=['fastrp', 'node2vec', 'transe', 'all'],
        help='Embedding method to train'
    )

    parser.add_argument(
        '--dim',
        type=int,
        default=128,
        help='Embedding dimension (default: 128)'
    )

    parser.add_argument(
        '--graph',
        type=str,
        default='data/metaqa/graph.pkl',
        help='Path to graph pickle file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='embeddings',
        help='Output directory (default: embeddings)'
    )

    # Node2Vec specific
    parser.add_argument(
        '--walks',
        type=int,
        default=200,
        help='Number of walks per node for Node2Vec (default: 200)'
    )

    parser.add_argument(
        '--walk-length',
        type=int,
        default=30,
        help='Walk length for Node2Vec (default: 30)'
    )

    parser.add_argument(
        '--checkpoints',
        type=int,
        nargs='+',
        help='Checkpoint values - Node2Vec: walk counts, FastRP: iterations, TransE: epoch counts (e.g., --checkpoints 10 50 100 200)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of parallel workers for Node2Vec (default: 8)'
    )

    # Training parameters (shared)
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Training epochs - Node2Vec: Word2Vec epochs, TransE: max epochs (default: 100)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=2048,
        help='Batch size for TransE (default: 2048)'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='Learning rate for TransE (default: 0.01)'
    )

    parser.add_argument(
        '--early-stopping',
        type=int,
        default=10,
        help='Early stopping patience - stops if no improvement for N epochs (default: 10)'
    )

    # FastRP specific
    parser.add_argument(
        '--iterations',
        type=int,
        default=5,
        help='Propagation iterations for FastRP (default: 5)'
    )

    parser.add_argument(
        '--iteration-checkpoints',
        type=int,
        nargs='+',
        help='FastRP checkpoint iteration counts (e.g., --iteration-checkpoints 1 2 3 5)'
    )

    args = parser.parse_args()

    # Setup logging
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"train_{args.method}_{timestamp}.log"

    logger = TimingLogger(str(log_file))

    logger.log("="*60)
    logger.log("🎯 MetaQA Graph Embedding Trainer")
    logger.log("="*60)
    logger.log(f"Method: {args.method}")
    logger.log(f"Dimension: {args.dim}")
    logger.log(f"Output: {args.output}")
    logger.log(f"Log file: {log_file}")
    logger.log("")

    # Initialize trainer
    trainer = EmbeddingTrainer(args.graph, logger)

    try:
        if args.method == 'fastrp' or args.method == 'all':
            results = trainer.train_fastrp(
                embedding_dim=args.dim,
                iterations=args.iterations,
                checkpoints=args.iteration_checkpoints
            )
            for embeddings, metadata in results:
                trainer.save_embeddings(embeddings, metadata, args.output)
            logger.log("")

        if args.method == 'node2vec' or args.method == 'all':
            if not NODE2VEC_AVAILABLE:
                logger.log("⚠️  node2vec not installed. Skipping.", level="WARNING")
            else:
                results = trainer.train_node2vec(
                    embedding_dim=args.dim,
                    walk_length=args.walk_length,
                    num_walks=args.walks,
                    workers=args.workers,
                    epochs=args.epochs,
                    checkpoints=args.checkpoints
                )
                for embeddings, metadata in results:
                    trainer.save_embeddings(embeddings, metadata, args.output)
                logger.log("")

        if args.method == 'transe' or args.method == 'all':
            if not PYKEEN_AVAILABLE:
                logger.log("⚠️  pykeen not installed. Skipping.", level="WARNING")
            else:
                results = trainer.train_transe(
                    embedding_dim=args.dim,
                    num_epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    patience=args.early_stopping,
                    checkpoints=args.checkpoints
                )
                for embeddings, metadata, relation_embeddings, _ in results:
                    trainer.save_embeddings(embeddings, metadata, args.output, relation_embeddings)
                logger.log("")

        logger.log("="*60)
        logger.log("✅ Training complete!")
        logger.log("="*60)

    except Exception as e:
        logger.log(f"❌ Training failed: {e}", level="ERROR")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
