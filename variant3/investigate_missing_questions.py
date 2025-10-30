"""
Investigate why metadata shows more questions than training data contains.
"""
import pickle
from pathlib import Path

from qa_system.config import Config
from qa_system.utils.loader import load_qa_dataset


def main():
    print("="*80)
    print("INVESTIGATING MISSING QUESTIONS")
    print("="*80)

    # Load training data
    training_data_path = Config.BASE_DIR / "data" / "variant3_training_data.pkl"

    print(f"\n[1] Loading training data...")
    with open(training_data_path, 'rb') as f:
        samples = pickle.load(f)

    print(f"   Total samples: {len(samples):,}")

    # Get unique questions from training data
    questions_in_training = set(sample['question_id'] for sample in samples)
    print(f"   Unique questions in training data: {len(questions_in_training):,}")

    # Load original questions
    print(f"\n[2] Loading original questions...")
    all_questions = []
    hop_counts = {}

    for hop_count in [1, 2, 3]:
        if hop_count == 1:
            qa_file = Config.QA_1HOP_TRAIN
        elif hop_count == 2:
            qa_file = Config.QA_2HOP_TRAIN
        else:
            qa_file = Config.QA_3HOP_TRAIN

        questions = load_qa_dataset(str(qa_file), hop_count=hop_count)
        print(f"   {hop_count}-hop: {len(questions):,} questions")
        all_questions.extend(questions)
        hop_counts[hop_count] = questions

    print(f"   Total original questions: {len(all_questions):,}")

    # Compare
    print(f"\n[3] Comparing...")

    original_question_ids = {q.question_id for q in all_questions}
    print(f"   Unique question IDs in original: {len(original_question_ids):,}")

    missing_from_training = original_question_ids - questions_in_training
    print(f"   Questions missing from training: {len(missing_from_training):,}")

    # Analyze missing questions by hop
    print(f"\n[4] Analyzing missing questions by hop count...")

    missing_by_hop = {1: 0, 2: 0, 3: 0}

    for hop_count, questions in hop_counts.items():
        for q in questions:
            if q.question_id in missing_from_training:
                missing_by_hop[hop_count] += 1

    print(f"\n   Missing questions by hop:")
    for hop in [1, 2, 3]:
        total_hop = len(hop_counts[hop])
        missing = missing_by_hop[hop]
        print(f"   {hop}-hop: {missing:,} / {total_hop:,} ({missing/total_hop*100:.1f}% missing)")

    # Show some examples
    print(f"\n[5] Sample missing questions...")

    samples_shown = 0
    for hop_count, questions in hop_counts.items():
        if samples_shown >= 10:
            break

        print(f"\n   --- {hop_count}-hop examples ---")
        hop_samples = 0

        for q in questions:
            if q.question_id in missing_from_training and hop_samples < 3:
                print(f"   • ID: {q.question_id}")
                print(f"     Text: {q.text[:100]}...")
                print(f"     Answers: {q.ground_truth_answers[:3]}")
                hop_samples += 1
                samples_shown += 1

            if samples_shown >= 10:
                break

    # Summary
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"""
Original questions:        {len(all_questions):>8,}
Questions in training:     {len(questions_in_training):>8,}
Missing from training:     {len(missing_from_training):>8,} ({len(missing_from_training)/len(all_questions)*100:.1f}%)

This matches the metadata:
- questions_with_path: 276,398 (but deduplicated to {len(questions_in_training):,})
- questions_without_path: 52,884

Wait... something's wrong with the metadata calculation!

Let me check: Are there duplicate question IDs?
""")

    # Check for duplicates
    question_id_counts = {}
    for q in all_questions:
        question_id_counts[q.question_id] = question_id_counts.get(q.question_id, 0) + 1

    duplicates = {qid: count for qid, count in question_id_counts.items() if count > 1}

    if duplicates:
        print(f"\n⚠️  Found {len(duplicates):,} duplicate question IDs!")
        print(f"   Showing first 10:")
        for i, (qid, count) in enumerate(list(duplicates.items())[:10], 1):
            print(f"   {i}. ID '{qid}' appears {count} times")
    else:
        print(f"\n✓ No duplicate question IDs found")

    # Check unique texts
    unique_texts = set(q.text for q in all_questions)
    print(f"\nUnique question texts: {len(unique_texts):,}")

    if len(unique_texts) != len(all_questions):
        print(f"⚠️  Some questions have identical text! This might be intentional.")


if __name__ == "__main__":
    main()