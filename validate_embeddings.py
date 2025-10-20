"""
Comprehensive validation script for graph embeddings.

Validates FastRP, Node2Vec, and TransE embeddings with:
- Universal sanity checks (cosine similarity, t-SNE/UMAP, nearest neighbors)
- Model-specific validation (structural, community, relational)
- Downstream validation (heuristic ranking, QA probe)

Usage:
    python validate_embeddings.py --embedding fastrp_embeddings_iter5.npy
    python validate_embeddings.py --embedding node2vec_embeddings_walks200.npy --method node2vec
    python validate_embeddings.py --embedding transe_embeddings.npy --method transe
    python validate_embeddings.py --all  # Validate all embeddings
"""

# Fix OpenBLAS threading issue
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import pickle
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import random
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Try to import UMAP
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class EmbeddingValidator:
    """Validate graph embeddings with comprehensive checks."""

    def __init__(self, graph_path: str = "data/metaqa/graph.pkl",
                 embedding_dir: str = "embeddings",
                 qa_path: str = "data/metaqa/1-hop/vanilla/qa_test.txt",
                 output_dir: str = "validation_results"):
        """Initialize validator."""
        print("="*70)
        print("🔍 Embedding Validation Suite")
        print("="*70)
        
        # Load graph
        print(f"\n📂 Loading graph from {graph_path}...")
        with open(graph_path, 'rb') as f:
            self.G = pickle.load(f)
        print(f"   ✓ Loaded: {self.G.number_of_nodes():,} nodes, {self.G.number_of_edges():,} edges")

        # Load node2id mapping
        node2id_path = Path(embedding_dir) / "node2id.json"
        print(f"\n📂 Loading node mapping from {node2id_path}...")
        with open(node2id_path, 'r') as f:
            self.node2id = json.load(f)
        self.id2node = {v: k for k, v in self.node2id.items()}
        print(f"   ✓ Loaded {len(self.node2id):,} node mappings")

        # Load QA data if available
        self.qa_data = []
        if Path(qa_path).exists():
            print(f"\n📂 Loading QA data from {qa_path}...")
            with open(qa_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        question = parts[0]
                        answers = parts[1].split('|')
                        self.qa_data.append((question, answers))
            print(f"   ✓ Loaded {len(self.qa_data):,} QA pairs")
        
        # Create output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.embedding_dir = Path(embedding_dir)

    def load_embedding(self, embedding_file: str) -> Tuple[np.ndarray, Dict]:
        """Load embedding and its metadata."""
        emb_path = self.embedding_dir / embedding_file
        emb = np.load(emb_path)
        
        # Load metadata
        meta_file = emb_path.stem.replace('_embeddings', '_metadata') + '.json'
        meta_path = self.embedding_dir / meta_file
        
        metadata = {}
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        
        return emb, metadata

    def cosine_similarity_check(self, emb: np.ndarray, name: str, use_l2: bool = False) -> Dict:
        """Check cosine similarity (or L2 distance for TransE) between connected vs random pairs."""
        metric_name = "L2 Distance" if use_l2 else "Cosine Similarity"
        print(f"\n{'='*70}")
        print(f"📊 {metric_name} Check: {name}")
        print(f"{'='*70}")

        if use_l2:
            print("   ℹ️  Using L2 distance (TransE mode) - lower is better for connected pairs")

        edges = list(self.G.edges())

        # Sample connected pairs
        n_samples = min(5000, len(edges))
        connected_pairs = random.sample(edges, n_samples)

        # Sample random pairs
        nodes = list(self.G.nodes())
        random_pairs = [(random.choice(nodes), random.choice(nodes))
                       for _ in range(n_samples)]

        def compute_avg_metric(pairs):
            metrics = []
            for a, b in pairs:
                if a in self.node2id and b in self.node2id:
                    idx_a = self.node2id[a]
                    idx_b = self.node2id[b]
                    if use_l2:
                        # L2 distance: lower means more similar
                        dist = np.linalg.norm(emb[idx_a] - emb[idx_b])
                        metrics.append(dist)
                    else:
                        # Cosine similarity: higher means more similar
                        sim = cosine_similarity(
                            emb[idx_a].reshape(1, -1),
                            emb[idx_b].reshape(1, -1)
                        )[0, 0]
                        metrics.append(sim)
            return np.mean(metrics) if metrics else 0.0

        connected_metric = compute_avg_metric(connected_pairs)
        random_metric = compute_avg_metric(random_pairs)

        if use_l2:
            print(f"\n   Connected pairs avg L2 distance: {connected_metric:.4f}")
            print(f"   Random pairs avg L2 distance:    {random_metric:.4f}")
            print(f"   Difference (random - connected): {random_metric - connected_metric:.4f}")

            # For L2 distance, connected should be LOWER (closer) than random
            if random_metric > connected_metric + 0.5:
                verdict = "✅ PASS - Clear separation between connected and random"
            elif random_metric > connected_metric:
                verdict = "⚠️  WEAK - Some separation but not strong"
            else:
                verdict = "❌ FAIL - No separation"
        else:
            print(f"\n   Connected pairs avg similarity: {connected_metric:.4f}")
            print(f"   Random pairs avg similarity:    {random_metric:.4f}")
            print(f"   Difference:                     {connected_metric - random_metric:.4f}")

            # For cosine similarity, connected should be HIGHER than random
            if connected_metric > random_metric + 0.2:
                verdict = "✅ PASS - Clear separation between connected and random"
            elif connected_metric > random_metric:
                verdict = "⚠️  WEAK - Some separation but not strong"
            else:
                verdict = "❌ FAIL - No separation"

        print(f"\n   {verdict}")

        return {
            'connected_similarity' if not use_l2 else 'connected_distance': float(connected_metric),
            'random_similarity' if not use_l2 else 'random_distance': float(random_metric),
            'difference': float(abs(connected_metric - random_metric)),
            'verdict': verdict,
            'metric': 'l2_distance' if use_l2 else 'cosine_similarity'
        }

    def visualize_embeddings(self, emb: np.ndarray, name: str, method: str = 'tsne'):
        """Create t-SNE or UMAP visualization."""
        print(f"\n{'='*70}")
        print(f"🎨 Creating {method.upper()} Visualization: {name}")
        print(f"{'='*70}")
        
        # Sample nodes for visualization
        n_samples = min(3000, emb.shape[0])
        indices = np.random.choice(emb.shape[0], n_samples, replace=False)
        X_sample = emb[indices]
        
        print(f"\n   Reducing {n_samples} nodes to 2D using {method.upper()}...")
        
        if method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
            Z = reducer.fit_transform(X_sample)
        elif method == 'umap' and UMAP_AVAILABLE:
            reducer = UMAP(n_components=2, random_state=42)
            Z = reducer.fit_transform(X_sample)
        else:
            print("   ⚠️  UMAP not available, falling back to PCA")
            reducer = PCA(n_components=2, random_state=42)
            Z = reducer.fit_transform(X_sample)
            method = 'pca'
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        scatter = ax.scatter(Z[:, 0], Z[:, 1], s=5, alpha=0.6, c=np.arange(len(Z)), cmap='viridis')
        ax.set_title(f"{method.upper()} Visualization: {name}", fontsize=14, fontweight='bold')
        ax.set_xlabel(f"{method.upper()}-1")
        ax.set_ylabel(f"{method.upper()}-2")
        plt.colorbar(scatter, ax=ax, label='Sample Index')
        
        # Save
        output_file = self.output_dir / f"{name}_{method}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved visualization: {output_file}")

    def nearest_neighbors_check(self, emb: np.ndarray, name: str, topk: int = 10, use_l2: bool = False):
        """Check nearest neighbors for a few entities."""
        metric_name = "L2 Distance" if use_l2 else "Cosine Similarity"
        print(f"\n{'='*70}")
        print(f"👥 Nearest Neighbors Check ({metric_name}): {name}")
        print(f"{'='*70}")

        if use_l2:
            print("   ℹ️  Using L2 distance (TransE mode) - lower values = closer neighbors")

        # Pick some test entities (actors, movies)
        test_entities = []
        nodes = list(self.G.nodes())

        # Try to find some known entities
        known_entities = ['Brad Pitt', 'Tom Hanks', 'Inception', 'The Matrix']
        for entity in known_entities:
            if entity in self.node2id:
                test_entities.append(entity)

        # If not enough, sample random
        if len(test_entities) < 3:
            test_entities = random.sample(nodes, min(5, len(nodes)))
        else:
            test_entities = test_entities[:3]

        results = {}
        for entity in test_entities:
            if entity not in self.node2id:
                continue

            idx = self.node2id[entity]

            if use_l2:
                # Compute L2 distances to all nodes
                dists = np.linalg.norm(emb - emb[idx], axis=1)
                top_indices = np.argsort(dists)[:topk]  # Sort ascending (lowest distance first)
                neighbors = [(self.id2node[i], float(dists[i])) for i in top_indices]
            else:
                # Compute cosine similarities to all nodes
                sims = cosine_similarity(emb[idx].reshape(1, -1), emb)[0]
                top_indices = np.argsort(-sims)[:topk]  # Sort descending (highest similarity first)
                neighbors = [(self.id2node[i], float(sims[i])) for i in top_indices]

            results[entity] = neighbors

            print(f"\n   Query: '{entity}'")
            print(f"   Top {topk} nearest neighbors:")
            for i, (neighbor, metric_val) in enumerate(neighbors, 1):
                # Highlight if same entity
                marker = "→" if neighbor == entity else " "
                metric_label = "dist" if use_l2 else "sim"
                print(f"      {i:2d}. {marker} {neighbor:<40} ({metric_label}: {metric_val:.4f})")

        return results

    def degree_correlation_check(self, emb: np.ndarray, name: str):
        """Check if node degree correlates with embedding position (FastRP)."""
        print(f"\n{'='*70}")
        print(f"📈 Degree Correlation Check (FastRP): {name}")
        print(f"{'='*70}")
        
        # Get node degrees
        degrees = dict(self.G.degree())
        deg_vals = np.array([degrees[self.id2node[i]] for i in range(emb.shape[0])])
        
        # Project to 2D using PCA
        pca = PCA(n_components=2, random_state=42)
        proj = pca.fit_transform(emb)
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        scatter = ax.scatter(proj[:, 0], proj[:, 1], c=deg_vals, s=5, alpha=0.6, cmap='viridis')
        ax.set_title(f"Degree Correlation: {name}", fontsize=14, fontweight='bold')
        ax.set_xlabel("PC-1")
        ax.set_ylabel("PC-2")
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Node Degree', rotation=270, labelpad=20)
        
        # Save
        output_file = self.output_dir / f"{name}_degree_correlation.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved visualization: {output_file}")
        
        # Compute correlation between degree and position
        corr_x = np.corrcoef(proj[:, 0], deg_vals)[0, 1]
        corr_y = np.corrcoef(proj[:, 1], deg_vals)[0, 1]
        
        print(f"\n   Correlation (PC-1 vs Degree): {corr_x:.4f}")
        print(f"   Correlation (PC-2 vs Degree): {corr_y:.4f}")
        
        if abs(corr_x) > 0.3 or abs(corr_y) > 0.3:
            print(f"   ✅ PASS - Degree information captured in embeddings")
        else:
            print(f"   ⚠️  WEAK - Limited degree information")
        
        return {'corr_pc1': float(corr_x), 'corr_pc2': float(corr_y)}

    def cluster_analysis(self, emb: np.ndarray, name: str, n_clusters: int = 20):
        """Analyze intra-cluster vs inter-cluster similarity (Node2Vec)."""
        print(f"\n{'='*70}")
        print(f"🔬 Cluster Analysis (Node2Vec): {name}")
        print(f"{'='*70}")
        
        # Perform clustering
        print(f"\n   Clustering into {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(emb)
        
        # Sample for efficiency
        n_samples = min(2000, emb.shape[0])
        indices = np.random.choice(emb.shape[0], n_samples, replace=False)
        
        X_sample = emb[indices]
        labels_sample = labels[indices]
        
        # Compute similarities
        sims = cosine_similarity(X_sample)
        
        intra_sims = []
        inter_sims = []
        
        for i in range(len(labels_sample)):
            for j in range(i+1, len(labels_sample)):
                if labels_sample[i] == labels_sample[j]:
                    intra_sims.append(sims[i, j])
                else:
                    inter_sims.append(sims[i, j])
        
        intra_mean = np.mean(intra_sims) if intra_sims else 0.0
        inter_mean = np.mean(inter_sims) if inter_sims else 0.0
        
        print(f"\n   Intra-cluster similarity: {intra_mean:.4f}")
        print(f"   Inter-cluster similarity: {inter_mean:.4f}")
        print(f"   Difference:              {intra_mean - inter_mean:.4f}")
        
        # Verdict
        if intra_mean > inter_mean + 0.2:
            verdict = "✅ PASS - Strong community structure"
        elif intra_mean > inter_mean:
            verdict = "⚠️  WEAK - Some community structure"
        else:
            verdict = "❌ FAIL - No community structure"
        
        print(f"\n   {verdict}")
        
        return {
            'intra_cluster_sim': float(intra_mean),
            'inter_cluster_sim': float(inter_mean),
            'difference': float(intra_mean - inter_mean),
            'verdict': verdict
        }

    def path_distance_correlation(self, emb: np.ndarray, name: str, use_l2: bool = False):
        """Check correlation between embedding distance and graph path length."""
        metric_name = "L2 Distance" if use_l2 else "Cosine-based Distance"
        print(f"\n{'='*70}")
        print(f"🛤️  Path Distance Correlation ({metric_name}): {name}")
        print(f"{'='*70}")

        if use_l2:
            print("   ℹ️  Using L2 distance (TransE mode)")

        # Sample source nodes
        nodes = list(self.G.nodes())
        n_sources = min(100, len(nodes))
        source_nodes = random.sample(nodes, n_sources)

        # Pick a target node
        target_node = random.choice(nodes)
        if target_node not in self.node2id:
            print("   ⚠️  Target node not in embedding, skipping...")
            return {}

        print(f"\n   Computing distances from {n_sources} nodes to '{target_node}'...")

        emb_dists = []
        path_lengths = []

        target_idx = self.node2id[target_node]
        target_emb = emb[target_idx]

        for source in source_nodes:
            if source not in self.node2id:
                continue

            # Check if path exists
            try:
                path_len = nx.shortest_path_length(self.G, source, target_node)
            except nx.NetworkXNoPath:
                continue

            # Compute embedding distance
            source_idx = self.node2id[source]
            source_emb = emb[source_idx]

            if use_l2:
                # Use L2 distance directly
                emb_dist = np.linalg.norm(source_emb - target_emb)
            else:
                # Use cosine-based distance
                cos_sim = cosine_similarity(source_emb.reshape(1, -1), target_emb.reshape(1, -1))[0, 0]
                emb_dist = 1 - cos_sim  # Convert similarity to distance

            emb_dists.append(emb_dist)
            path_lengths.append(path_len)

        if len(emb_dists) < 10:
            print("   ⚠️  Not enough valid paths, skipping...")
            return {}

        # Compute correlation
        correlation = np.corrcoef(emb_dists, path_lengths)[0, 1]

        print(f"\n   Valid paths found: {len(emb_dists)}")
        print(f"   Correlation (emb_dist vs path_length): {correlation:.4f}")

        # Create scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.scatter(path_lengths, emb_dists, alpha=0.5, s=20)
        ax.set_xlabel('Graph Path Length (hops)')
        ylabel = 'Embedding Distance (L2)' if use_l2 else 'Embedding Distance (1 - cosine_sim)'
        ax.set_ylabel(ylabel)
        ax.set_title(f"Path Distance Correlation: {name}\nCorrelation: {correlation:.4f}",
                    fontsize=12, fontweight='bold')

        # Add trend line
        z = np.polyfit(path_lengths, emb_dists, 1)
        p = np.poly1d(z)
        ax.plot(sorted(set(path_lengths)), p(sorted(set(path_lengths))), "r--", alpha=0.8, linewidth=2)

        # Save
        output_file = self.output_dir / f"{name}_path_correlation.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"   ✓ Saved visualization: {output_file}")

        # Verdict
        if correlation > 0.5:
            verdict = "✅ PASS - Strong correlation"
        elif correlation > 0.3:
            verdict = "⚠️  WEAK - Moderate correlation"
        else:
            verdict = "❌ FAIL - Weak/no correlation"

        print(f"\n   {verdict}")

        return {
            'correlation': float(correlation),
            'n_samples': len(emb_dists),
            'verdict': verdict,
            'metric': 'l2_distance' if use_l2 else 'cosine_distance'
        }

    def qa_probe_test(self, emb: np.ndarray, name: str, topk: int = 10, use_l2: bool = False):
        """Test QA retrieval accuracy using embeddings."""
        metric_name = "L2 Distance" if use_l2 else "Cosine Similarity"
        print(f"\n{'='*70}")
        print(f"❓ QA Probe Test ({metric_name}): {name}")
        print(f"{'='*70}")

        if use_l2:
            print("   ℹ️  Using L2 distance (TransE mode)")

        if not self.qa_data:
            print("   ⚠️  No QA data available, skipping...")
            return {}

        # Sample QA pairs
        n_samples = min(100, len(self.qa_data))
        qa_sample = random.sample(self.qa_data, n_samples)

        hits = 0
        valid_questions = 0

        print(f"\n   Testing {n_samples} QA pairs...")

        for question, answers in qa_sample:
            # Extract entity from question (between brackets)
            import re
            entities = re.findall(r'\[(.*?)\]', question)
            if not entities:
                continue

            query_entity = entities[0]
            if query_entity not in self.node2id:
                continue

            valid_questions += 1

            # Get nearest neighbors
            idx = self.node2id[query_entity]

            if use_l2:
                # Use L2 distance - lower is better
                dists = np.linalg.norm(emb - emb[idx], axis=1)
                top_indices = np.argsort(dists)[:topk]  # Sort ascending (lowest distance first)
            else:
                # Use cosine similarity - higher is better
                sims = cosine_similarity(emb[idx].reshape(1, -1), emb)[0]
                top_indices = np.argsort(-sims)[:topk]  # Sort descending (highest similarity first)

            neighbors = [self.id2node[i] for i in top_indices]

            # Check if any answer is in top-k
            if any(ans in neighbors for ans in answers):
                hits += 1

        if valid_questions == 0:
            print("   ⚠️  No valid questions found, skipping...")
            return {}

        accuracy = hits / valid_questions

        print(f"\n   Valid questions: {valid_questions}")
        print(f"   Hits@{topk}:        {hits} ({accuracy*100:.1f}%)")

        # Verdict
        if accuracy >= 0.6:
            verdict = "✅ PASS - Good QA performance"
        elif accuracy >= 0.4:
            verdict = "⚠️  WEAK - Moderate QA performance"
        else:
            verdict = "❌ FAIL - Poor QA performance"

        print(f"\n   {verdict}")

        return {
            'valid_questions': valid_questions,
            'hits': hits,
            'topk': topk,
            'accuracy': float(accuracy),
            'verdict': verdict,
            'metric': 'l2_distance' if use_l2 else 'cosine_similarity'
        }

    def transe_link_prediction_check(self, emb: np.ndarray, name: str):
        """TransE-specific link prediction validation using L2 distance."""
        print(f"\n{'='*70}")
        print(f"🔗 TransE Link Prediction Check: {name}")
        print(f"{'='*70}")

        # Try to load relation embeddings
        base_name = name.replace('_embeddings', '')
        rel_emb_file = self.embedding_dir / f"{base_name}_relation_embeddings.npy"

        if not rel_emb_file.exists():
            print(f"\n   ⚠️  Relation embeddings not found at {rel_emb_file}")
            print(f"   Cannot perform TransE-specific validation without relation embeddings.")
            print(f"   Note: TransE should be evaluated with L2 distance, not cosine similarity!")
            return {'error': 'relation_embeddings_not_found'}

        rel_emb = np.load(rel_emb_file)
        print(f"\n   ✓ Loaded relation embeddings: {rel_emb.shape}")

        # Sample edges for evaluation
        edges = list(self.G.edges(data=True))
        n_samples = min(1000, len(edges))
        test_edges = random.sample(edges, n_samples)

        # For each edge, compute TransE score: score = -||h + r - t||
        scores_connected = []
        scores_random = []

        print(f"\n   Computing TransE scores for {n_samples} edges...")

        for h, t, data in test_edges:
            if h not in self.node2id or t not in self.node2id:
                continue

            relation = data.get('relation', 'related_to')

            # Get embeddings
            h_idx = self.node2id[h]
            t_idx = self.node2id[t]
            h_emb = emb[h_idx]
            t_emb = emb[t_idx]

            # We need relation ID - for now use first relation as proxy
            # In proper implementation, would map relation string to ID
            r_emb = rel_emb[0]  # Simplified: use first relation

            # TransE score: -||h + r - t||
            score = -np.linalg.norm(h_emb + r_emb - t_emb)
            scores_connected.append(score)

            # Random pair
            random_t = random.choice(list(self.node2id.keys()))
            random_t_idx = self.node2id[random_t]
            random_t_emb = emb[random_t_idx]
            random_score = -np.linalg.norm(h_emb + r_emb - random_t_emb)
            scores_random.append(random_score)

        if not scores_connected:
            print(f"   ⚠️  No valid edges found for evaluation")
            return {}

        connected_mean = np.mean(scores_connected)
        random_mean = np.mean(scores_random)

        print(f"\n   Connected edges avg score: {connected_mean:.4f}")
        print(f"   Random pairs avg score:    {random_mean:.4f}")
        print(f"   Difference (higher=better): {connected_mean - random_mean:.4f}")

        # For TransE, connected edges should have HIGHER scores (less negative)
        if connected_mean > random_mean + 0.5:
            verdict = "✅ PASS - TransE properly distinguishes edges"
        elif connected_mean > random_mean:
            verdict = "⚠️  WEAK - Some separation but not strong"
        else:
            verdict = "❌ FAIL - No separation"

        print(f"\n   {verdict}")

        return {
            'connected_score': float(connected_mean),
            'random_score': float(random_mean),
            'difference': float(connected_mean - random_mean),
            'verdict': verdict,
            'note': 'TransE uses L2 distance, not cosine similarity'
        }

    def validate_embedding(self, embedding_file: str, method: str = None):
        """Run full validation suite on an embedding."""
        print(f"\n\n{'#'*70}")
        print(f"# VALIDATING: {embedding_file}")
        print(f"{'#'*70}")
        
        # Load embedding
        emb, metadata = self.load_embedding(embedding_file)
        
        # Infer method from metadata or filename
        if method is None:
            if metadata and 'method' in metadata:
                method = metadata['method'].lower()
            else:
                if 'fastrp' in embedding_file.lower():
                    method = 'fastrp'
                elif 'node2vec' in embedding_file.lower():
                    method = 'node2vec'
                elif 'transe' in embedding_file.lower():
                    method = 'transe'
                else:
                    method = 'unknown'
        
        name = embedding_file.replace('.npy', '')
        
        print(f"\n   Method: {method.upper()}")
        print(f"   Shape: {emb.shape}")
        print(f"   Metadata: {metadata if metadata else 'Not available'}")
        
        # Results dictionary
        results = {
            'embedding_file': embedding_file,
            'method': method,
            'shape': emb.shape,
            'metadata': metadata,
            'checks': {}
        }
        
        # Universal checks
        print(f"\n{'='*70}")
        print("📋 UNIVERSAL VALIDATION CHECKS")
        print(f"{'='*70}")
        
        results['checks']['cosine_similarity'] = self.cosine_similarity_check(emb, name)
        self.visualize_embeddings(emb, name, method='tsne')
        if UMAP_AVAILABLE:
            self.visualize_embeddings(emb, name, method='umap')
        results['checks']['nearest_neighbors'] = self.nearest_neighbors_check(emb, name)
        results['checks']['path_correlation'] = self.path_distance_correlation(emb, name)
        results['checks']['qa_probe'] = self.qa_probe_test(emb, name)
        
        # Model-specific checks
        print(f"\n{'='*70}")
        print(f"📋 MODEL-SPECIFIC CHECKS ({method.upper()})")
        print(f"{'='*70}")
        
        if method == 'fastrp':
            results['checks']['degree_correlation'] = self.degree_correlation_check(emb, name)

        if method == 'node2vec':
            results['checks']['cluster_analysis'] = self.cluster_analysis(emb, name)

        if method == 'transe':
            results['checks']['transe_link_prediction'] = self.transe_link_prediction_check(emb, name)

        # Save results
        results_file = self.output_dir / f"{name}_validation.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"✅ Validation complete for {embedding_file}")
        print(f"   Results saved: {results_file}")
        print(f"{'='*70}")
        
        return results

    def validate_all(self):
        """Validate all embeddings in the embeddings directory."""
        embedding_files = sorted(self.embedding_dir.glob("*_embeddings*.npy"))
        
        print(f"\n\nFound {len(embedding_files)} embeddings to validate:")
        for f in embedding_files:
            print(f"   - {f.name}")
        
        all_results = []
        for emb_file in embedding_files:
            results = self.validate_embedding(emb_file.name)
            all_results.append(results)
        
        # Create summary report
        self.create_summary_report(all_results)
        
        return all_results

    def create_summary_report(self, all_results: List[Dict]):
        """Create a summary report comparing all embeddings."""
        print(f"\n\n{'='*70}")
        print("📊 SUMMARY REPORT")
        print(f"{'='*70}")
        
        # Create summary table
        summary = []
        for result in all_results:
            name = result['embedding_file'].replace('_embeddings.npy', '').replace('_embeddings_', '_')
            method = result['method']
            
            row = {
                'Embedding': name,
                'Method': method.upper(),
                'Dim': result['shape'][1] if len(result['shape']) > 1 else result['shape'][0]
            }
            
            # Add check results
            checks = result['checks']
            if 'cosine_similarity' in checks:
                row['Conn_Sim'] = f"{checks['cosine_similarity']['connected_similarity']:.3f}"
                row['Rand_Sim'] = f"{checks['cosine_similarity']['random_similarity']:.3f}"
            
            if 'path_correlation' in checks and 'correlation' in checks['path_correlation']:
                row['Path_Corr'] = f"{checks['path_correlation']['correlation']:.3f}"
            
            if 'qa_probe' in checks and 'accuracy' in checks['qa_probe']:
                row['QA_Acc'] = f"{checks['qa_probe']['accuracy']*100:.1f}%"
            
            if 'cluster_analysis' in checks:
                row['Intra_Sim'] = f"{checks['cluster_analysis']['intra_cluster_sim']:.3f}"
            
            if 'degree_correlation' in checks:
                row['Deg_Corr'] = f"{checks['degree_correlation']['corr_pc1']:.3f}"
            
            summary.append(row)
        
        # Print table
        if summary:
            print("\n")
            headers = list(summary[0].keys())
            col_widths = {h: max(len(h), max(len(str(row.get(h, ''))) for row in summary)) for h in headers}
            
            # Print header
            header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
            print(header_line)
            print("-" * len(header_line))
            
            # Print rows
            for row in summary:
                print(" | ".join(str(row.get(h, '')).ljust(col_widths[h]) for h in headers))
        
        # Save summary
        summary_file = self.output_dir / "validation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n✓ Summary saved: {summary_file}")
        
        # Print recommendations
        print(f"\n{'='*70}")
        print("💡 RECOMMENDATIONS")
        print(f"{'='*70}")
        
        # Find best embedding for different tasks
        best_qa = max(all_results, 
                     key=lambda x: x['checks'].get('qa_probe', {}).get('accuracy', 0))
        best_path = max(all_results,
                       key=lambda x: x['checks'].get('path_correlation', {}).get('correlation', 0))
        
        print(f"\n   Best for QA:        {best_qa['embedding_file']}")
        if 'qa_probe' in best_qa['checks'] and 'accuracy' in best_qa['checks']['qa_probe']:
            print(f"                       (accuracy: {best_qa['checks']['qa_probe']['accuracy']*100:.1f}%)")
        
        print(f"\n   Best for Pathfinding: {best_path['embedding_file']}")
        if 'path_correlation' in best_path['checks'] and 'correlation' in best_path['checks']['path_correlation']:
            print(f"                         (correlation: {best_path['checks']['path_correlation']['correlation']:.3f})")


def main():
    parser = argparse.ArgumentParser(
        description="Validate graph embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--embedding',
        type=str,
        help='Specific embedding file to validate (e.g., fastrp_embeddings_iter5.npy)'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['fastrp', 'node2vec', 'transe'],
        help='Embedding method (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Validate all embeddings in the embeddings directory'
    )
    
    parser.add_argument(
        '--graph',
        type=str,
        default='data/metaqa/graph.pkl',
        help='Path to graph pickle file'
    )
    
    parser.add_argument(
        '--embedding-dir',
        type=str,
        default='embeddings',
        help='Directory containing embeddings'
    )
    
    parser.add_argument(
        '--qa-path',
        type=str,
        default='data/metaqa/1-hop/vanilla/qa_test.txt',
        help='Path to QA test file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='validation_results',
        help='Output directory for validation results'
    )
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = EmbeddingValidator(
        graph_path=args.graph,
        embedding_dir=args.embedding_dir,
        qa_path=args.qa_path,
        output_dir=args.output
    )
    
    # Run validation
    if args.all:
        validator.validate_all()
    elif args.embedding:
        validator.validate_embedding(args.embedding, args.method)
    else:
        print("\n❌ Error: Please specify --embedding or --all")
        parser.print_help()
        return 1
    
    print(f"\n\n{'='*70}")
    print("✅ VALIDATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\nResults saved to: {args.output}/")
    print("Check the validation_summary.json for a complete overview.")
    
    return 0


if __name__ == "__main__":
    exit(main())

