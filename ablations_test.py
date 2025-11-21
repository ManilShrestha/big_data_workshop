#!/usr/bin/env python3
"""
Unified Ablation Evaluation Script

Evaluates trained ablation models on QA datasets using beam search.

Ablation configurations:
- abl1: Text Embeddings + Hop Count (no graph)
- abl2: Text Embeddings only (no hop, no graph)
- abl3: Graph Embeddings + Hop Count (no text)
- abl4: Graph Embeddings only (no hop, no text)

Usage:
  python ablations_test.py --ablation abl1 --datasets 1-hop 2-hop 3-hop
  python ablations_test.py --ablation abl2 --datasets 1-hop --limit 100
  python ablations_test.py --ablation all --datasets 1-hop 2-hop 3-hop
"""

import argparse
import pickle
import json
import numpy as np
import torch
import time
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Set
from collections import deque
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from qa_system.config import Config
from qa_system.entity_linkers.exact_matcher import ExactMatcher
from qa_system.utils.loader import load_graph, load_node2id, load_qa_dataset

# Import all model classes
from variant3.variant3_edge_scorer_abl3_text_hops import EdgeScorerTextHops  # abl1
from variant3.variant3_edge_scorer_abl1_text_only import EdgeScorerTextOnly  # abl2
from variant3.variant3_edge_scorer_abl3_graph_hops import EdgeScorerGraphHops  # abl3
from variant3.variant3_edge_scorer_abl2_graph_only import EdgeScorerGraphOnly  # abl4

# Ablation configurations (same as train)
ABLATION_CONFIG = {
    'abl1': {
        'name': 'Text + Hop Count',
        'model_class': EdgeScorerTextHops,
        'needs_text': True,
        'needs_graph': False,
        'needs_hop': True,
        'model_file': 'ablation1_text_hops_best.pt',
    },
    'abl2': {
        'name': 'Text Only',
        'model_class': EdgeScorerTextOnly,
        'needs_text': True,
        'needs_graph': False,
        'needs_hop': False,
        'model_file': 'ablation2_text_only_best.pt',
    },
    'abl3': {
        'name': 'Graph + Hop Count',
        'model_class': EdgeScorerGraphHops,
        'needs_text': False,
        'needs_graph': True,
        'needs_hop': True,
        'model_file': 'ablation3_graph_hops_best.pt',
    },
    'abl4': {
        'name': 'Graph Only',
        'model_class': EdgeScorerGraphOnly,
        'needs_text': False,
        'needs_graph': True,
        'needs_hop': False,
        'model_file': 'ablation4_graph_only_best.pt',
    },
}


class AblationEdgeScorer:
    """Edge scorer for ablation models"""

    def __init__(self, ablation_name: str, device: str = None):
        self.config = ABLATION_CONFIG[ablation_name]
        self.ablation_name = ablation_name

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"[{ablation_name}] Device: {self.device}")

        # Load model
        model_path = Config.BASE_DIR / "models" / self.config['model_file']
        print(f"[{ablation_name}] Loading model from {model_path}...")

        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        ModelClass = self.config['model_class']
        if self.config['needs_text'] and self.config['needs_hop']:
            self.model = ModelClass(text_dim=1536, hidden_dim=512, max_hops=3, dropout=0.3)
        elif self.config['needs_text'] and not self.config['needs_hop']:
            self.model = ModelClass(text_dim=1536, hidden_dim=512, dropout=0.3)
        elif self.config['needs_graph'] and self.config['needs_hop']:
            self.model = ModelClass(graph_dim=256, hidden_dim=512, max_hops=3, dropout=0.3)
        elif self.config['needs_graph'] and not self.config['needs_hop']:
            self.model = ModelClass(graph_dim=256, hidden_dim=512, dropout=0.3)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Load embeddings
        self._load_embeddings()

        # OpenAI client for embedding new questions (text-based ablations)
        if self.config['needs_text']:
            self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
            self.question_cache = {}
            self.total_cost = 0.0

        self.zero_text = np.zeros(1536, dtype=np.float32)
        self.zero_graph = np.zeros(256, dtype=np.float32)

        print(f"[{ablation_name}] Initialization complete!")

    def _load_embeddings(self):
        """Load required embeddings based on config"""
        emb_dir = Config.EMBEDDINGS_DIR

        if self.config['needs_text']:
            print(f"   Loading text embeddings...")
            with open(emb_dir / "variant3_text_embeddings_cache.pkl", 'rb') as f:
                self.text_embeddings_cache = pickle.load(f)
            print(f"      {len(self.text_embeddings_cache):,} unique texts")

        if self.config['needs_graph']:
            print(f"   Loading graph embeddings...")
            self.entity_graph_embeddings = np.load(Config.TRANSE_ENTITY_PATH)
            self.relation_graph_embeddings = np.load(Config.TRANSE_RELATION_PATH)

            with open(emb_dir / "node2id.json", 'r') as f:
                self.entity2id = json.load(f)

            relation2id_path = emb_dir / "relation2id.json"
            if relation2id_path.exists():
                with open(relation2id_path, 'r') as f:
                    self.relation2id = json.load(f)
            else:
                with open(Config.TRANSE_METADATA_PATH, 'r') as f:
                    meta = json.load(f)
                self.relation2id = meta.get('relation_to_id', meta.get('relation2id', {}))

            print(f"      Entities: {self.entity_graph_embeddings.shape}")
            print(f"      Relations: {self.relation_graph_embeddings.shape}")

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Get text embedding"""
        if not self.config['needs_text']:
            return self.zero_text

        if text in self.text_embeddings_cache:
            return self.text_embeddings_cache[text]

        if text in self.question_cache:
            return self.question_cache[text]

        # Embed via API
        response = self.openai_client.embeddings.create(
            model=Config.OPENAI_MODEL_EMBED,
            input=[text]
        )
        emb = np.array(response.data[0].embedding, dtype=np.float32)
        self.question_cache[text] = emb

        # Track cost
        tokens = response.usage.total_tokens
        self.total_cost += tokens * 0.02 / 1_000_000  # text-embedding-3-small price

        return emb

    def _get_entity_embedding(self, entity_id: str) -> np.ndarray:
        """Get graph embedding for entity"""
        if not self.config['needs_graph']:
            return self.zero_graph

        if entity_id in self.entity2id:
            idx = self.entity2id[entity_id]
            return self.entity_graph_embeddings[idx]
        return self.zero_graph

    def _get_relation_embedding(self, relation: str) -> np.ndarray:
        """Get graph embedding for relation"""
        if not self.config['needs_graph']:
            return self.zero_graph

        if relation in self.relation2id:
            idx = self.relation2id[relation]
            return self.relation_graph_embeddings[idx]
        return self.zero_graph

    def score_edges(
        self,
        question_text: str,
        current_node: str,
        edges: List[Tuple[str, str, str]],  # (relation, target, direction)
        current_hop: int
    ) -> List[Tuple[str, str, str, float]]:
        """
        Score a batch of edges.

        Args:
            question_text: Question text
            current_node: Current node ID
            edges: List of (relation, target, direction) tuples
            current_hop: Current hop (0-indexed)

        Returns:
            List of (relation, target, direction, score) tuples
        """
        if not edges:
            return []

        batch_size = len(edges)

        # Build tensors based on config
        with torch.no_grad():
            if self.config['needs_text']:
                q_text = torch.tensor(self._get_text_embedding(question_text), dtype=torch.float32).unsqueeze(0).expand(batch_size, -1).to(self.device)
                n_text = torch.tensor(self._get_text_embedding(current_node), dtype=torch.float32).unsqueeze(0).expand(batch_size, -1).to(self.device)

                e_text_list = [self._get_text_embedding(rel) for rel, _, _ in edges]
                e_text = torch.tensor(np.stack(e_text_list), dtype=torch.float32).to(self.device)

                t_text_list = [self._get_text_embedding(target) for _, target, _ in edges]
                t_text = torch.tensor(np.stack(t_text_list), dtype=torch.float32).to(self.device)

            if self.config['needs_graph']:
                n_graph = torch.tensor(self._get_entity_embedding(current_node), dtype=torch.float32).unsqueeze(0).expand(batch_size, -1).to(self.device)

                e_graph_list = [self._get_relation_embedding(rel) for rel, _, _ in edges]
                e_graph = torch.tensor(np.stack(e_graph_list), dtype=torch.float32).to(self.device)

                t_graph_list = [self._get_entity_embedding(target) for _, target, _ in edges]
                t_graph = torch.tensor(np.stack(t_graph_list), dtype=torch.float32).to(self.device)

            if self.config['needs_hop']:
                hop = torch.tensor([current_hop] * batch_size, dtype=torch.long).to(self.device)

            # Forward pass
            if self.config['needs_text'] and self.config['needs_hop']:
                logits = self.model(q_text, n_text, e_text, t_text, hop)
            elif self.config['needs_text'] and not self.config['needs_hop']:
                logits = self.model(q_text, n_text, e_text, t_text)
            elif self.config['needs_graph'] and self.config['needs_hop']:
                logits = self.model(n_graph, e_graph, t_graph, hop)
            elif self.config['needs_graph'] and not self.config['needs_hop']:
                logits = self.model(n_graph, e_graph, t_graph)

            scores = torch.sigmoid(logits).cpu().numpy().flatten()

        # Return scored edges
        return [(rel, target, direction, float(scores[i])) for i, (rel, target, direction) in enumerate(edges)]


def beam_search(
    graph,
    scorer: AblationEdgeScorer,
    question_text: str,
    start_nodes: List[str],
    max_hops: int,
    beam_width: int = 10,
    top_k_relations: int = 3
) -> Tuple[List[str], int]:
    """
    Beam search guided by ablation edge scorer.

    Returns:
        (answers, nodes_expanded)
    """
    nodes_expanded = 0
    current_beam = [(node, 0, [node]) for node in start_nodes if node in graph]

    if not current_beam:
        return [], 0

    for hop in range(max_hops):
        candidates = []

        for node, _, path in current_beam:
            if node not in graph:
                continue

            nodes_expanded += 1
            edges = []

            # Collect all edges
            for pred, _, key, data in graph.in_edges(node, keys=True, data=True):
                if pred not in path:  # Avoid cycles
                    rel = data.get('relation', '')
                    edges.append((rel, pred, 'in'))

            for _, succ, key, data in graph.out_edges(node, keys=True, data=True):
                if succ not in path:
                    rel = data.get('relation', '')
                    edges.append((rel, succ, 'out'))

            if not edges:
                continue

            # Score edges
            scored_edges = scorer.score_edges(question_text, node, edges, hop)

            # Sort by score and take top
            scored_edges.sort(key=lambda x: x[3], reverse=True)

            for rel, target, direction, score in scored_edges[:beam_width]:
                new_path = path + [target]
                candidates.append((target, score, new_path))

        if not candidates:
            break

        # Keep top beam_width candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        current_beam = candidates[:beam_width]

    # Extract final answers
    answers = [node for node, _, _ in current_beam]
    return list(set(answers)), nodes_expanded


def evaluate_ablation(
    ablation_name: str,
    datasets: List[str],
    limit: int = None,
    beam_width: int = 10,
    device: str = None
):
    """Evaluate an ablation on specified datasets"""

    print("\n" + "=" * 80)
    print(f"Evaluating {ablation_name.upper()}: {ABLATION_CONFIG[ablation_name]['name']}")
    print("=" * 80)

    # Load graph
    print("\n[1/3] Loading graph...")
    graph = load_graph(Config.GRAPH_PATH)
    node2id = load_node2id(Config.NODE2ID_PATH)

    # Initialize entity linker
    print("\n[2/3] Initializing components...")
    entity_linker = ExactMatcher(node2id)

    # Initialize scorer
    scorer = AblationEdgeScorer(ablation_name, device=device)

    # Dataset mapping
    dataset_map = {
        '1-hop': (Config.QA_1HOP_TEST, 1),
        '2-hop': (Config.QA_2HOP_TEST, 2),
        '3-hop': (Config.QA_3HOP_TEST, 3),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}

    for dataset_name in datasets:
        print(f"\n[3/3] Evaluating on {dataset_name}...")
        print("-" * 80)

        dataset_path, hop_count = dataset_map[dataset_name]
        questions = load_qa_dataset(dataset_path, hop_count=hop_count, limit=limit)

        correct = 0
        total = 0
        total_nodes = 0
        total_time = 0

        all_preds = []
        all_gts = []

        for q in tqdm(questions, desc=f"{dataset_name}"):
            start_time = time.time()

            # Entity linking
            start_nodes = entity_linker.extract_and_link(q.text)

            if not start_nodes:
                predicted = []
            else:
                predicted, nodes_expanded = beam_search(
                    graph, scorer, q.text, start_nodes, hop_count, beam_width
                )
                total_nodes += nodes_expanded

            elapsed = (time.time() - start_time) * 1000
            total_time += elapsed

            # Compute accuracy
            gt_set = set(q.ground_truth_answers)
            pred_set = set(predicted)

            all_preds.append(pred_set)
            all_gts.append(gt_set)

            if pred_set & gt_set:  # Any overlap is correct
                correct += 1
            total += 1

        # Compute metrics
        accuracy = correct / total if total > 0 else 0
        avg_nodes = total_nodes / total if total > 0 else 0
        avg_time = total_time / total if total > 0 else 0

        # Compute micro F1
        tp = sum(len(p & g) for p, g in zip(all_preds, all_gts))
        fp = sum(len(p - g) for p, g in zip(all_preds, all_gts))
        fn = sum(len(g - p) for p, g in zip(all_preds, all_gts))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_nodes_expanded': avg_nodes,
            'avg_time_ms': avg_time,
            'total_questions': total,
        }

        all_results[dataset_name] = results

        print(f"\n{dataset_name} Results:")
        print(f"  Accuracy:    {accuracy:.2%}")
        print(f"  Precision:   {precision:.2%}")
        print(f"  Recall:      {recall:.2%}")
        print(f"  F1:          {f1:.2%}")
        print(f"  Avg nodes:   {avg_nodes:.1f}")
        print(f"  Avg time:    {avg_time:.1f} ms")

    # Save results
    results_path = Config.BASE_DIR / "results" / f"{ablation_name}_eval_{timestamp}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, 'w') as f:
        json.dump({
            'ablation': ablation_name,
            'ablation_name': ABLATION_CONFIG[ablation_name]['name'],
            'datasets': all_results,
            'beam_width': beam_width,
            'limit': limit,
            'timestamp': timestamp
        }, f, indent=2)

    print(f"\nResults saved: {results_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate ablation models on QA datasets')
    parser.add_argument('--ablation', type=str, required=True,
                       choices=['abl1', 'abl2', 'abl3', 'abl4', 'all'],
                       help='Which ablation to evaluate')
    parser.add_argument('--datasets', type=str, nargs='+',
                       choices=['1-hop', '2-hop', '3-hop'],
                       default=['1-hop'],
                       help='Datasets to evaluate on')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit questions per dataset')
    parser.add_argument('--beam-width', type=int, default=10,
                       help='Beam width for search')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'auto'],
                       default='auto', help='Device for inference')

    args = parser.parse_args()

    device = None if args.device == 'auto' else args.device

    if args.ablation == 'all':
        all_ablation_results = {}
        for abl in ['abl1', 'abl2', 'abl3', 'abl4']:
            try:
                results = evaluate_ablation(
                    abl, args.datasets, args.limit, args.beam_width, device
                )
                all_ablation_results[abl] = results
            except FileNotFoundError as e:
                print(f"\nSkipping {abl}: Model not found ({e})")

        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY: All Ablations")
        print("=" * 80)

        for abl, results in all_ablation_results.items():
            config = ABLATION_CONFIG[abl]
            print(f"\n{abl.upper()}: {config['name']}")
            for ds, metrics in results.items():
                print(f"  {ds}: Acc={metrics['accuracy']:.2%}, F1={metrics['f1']:.2%}")
    else:
        evaluate_ablation(args.ablation, args.datasets, args.limit, args.beam_width, device)


if __name__ == "__main__":
    main()
