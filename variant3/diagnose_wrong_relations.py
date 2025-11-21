#!/usr/bin/env python3
"""
Detailed analysis of WHY wrong relations are selected in 3-hop failures.
We'll manually trace through a few failure cases to see what the scorer is doing.
"""

import json
from pathlib import Path

def main():
    # Load results
    results_dir = Path(__file__).parent.parent / 'results'
    result_files = sorted(results_dir.glob('variant3_edge_scorer_3-hop*.json'))
    result_file = result_files[-1]

    with open(result_file, 'r') as f:
        data = json.load(f)

    # Find failures
    failures = [r for r in data['results'] if not r['is_correct']]

    print("=" * 80)
    print("DETAILED RELATION ANALYSIS")
    print("=" * 80)

    # Analyze first 3 failures in detail
    for idx, failure in enumerate(failures[:3]):
        print(f"\n{'=' * 80}")
        print(f"FAILURE {idx + 1}")
        print(f"{'=' * 80}")
        print(f"Question: {failure['question_text']}")
        print(f"\nGround truth: {failure['ground_truth_answers']}")
        print(f"Predicted: {failure['predicted_answers']}")
        print(f"\nRelations used: {failure['relations_used']}")

        # Analyze metadata if available
        metadata = failure.get('metadata', {})
        if 'start_entity' in metadata:
            print(f"\nStart entity: {metadata['start_entity']}")

        print("\nQuestion analysis:")
        question_lower = failure['question_text'].lower()

        # Identify question intent
        if 'director' in question_lower:
            print("  - Asking about DIRECTORS")
        if 'written' in question_lower or 'writer' in question_lower:
            print("  - Asking about WRITERS")
        if 'language' in question_lower:
            print("  - Asking about LANGUAGES")
        if 'actor' in question_lower or 'acted' in question_lower:
            print("  - Asking about ACTORS")
        if 'release' in question_lower:
            print("  - Asking about RELEASE YEARS")

        print("\nRelation path analysis:")
        relations = failure['relations_used']
        for hop_idx, rel in enumerate(relations):
            print(f"  Hop {hop_idx + 1}: {rel}")

            # Check if relation matches question intent
            if 'director' in question_lower and 'directed' not in rel:
                print(f"      Question asks about directors but used: {rel}")
            if ('written' in question_lower or 'writer' in question_lower) and 'written' not in rel:
                print(f"      Question asks about writers but used: {rel}")
            if 'language' in question_lower and hop_idx == len(relations) - 1 and 'language' not in rel:
                print(f"      Question asks about language but final hop is: {rel}")

        print()

if __name__ == '__main__':
    main()
