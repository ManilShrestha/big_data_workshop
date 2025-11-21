"""
Phase 0: Test LLM Decomposition on Sample Questions

This script tests the LLM prompt on a small sample of questions to verify:
1. LLM correctly uses [answer_from_hop_N] placeholders
2. LLM picks reasonable relations from METAQA_RELATIONS
3. Output format is correct and parseable

We manually review 15 samples (5 from each hop count) before proceeding
to full-scale decomposition in Phase 1.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from qa_system.config import Config
from qa_system.utils.loader import load_qa_dataset

# MetaQA Relations
METAQA_RELATIONS = [
    "directed_by",
    "starred_actors",
    "written_by",
    "in_language",
    "has_genre",
    "release_year",
    "has_tags",
    "has_imdb_rating",
    "has_imdb_votes"
]


def build_decomposition_prompt(question_text: str, hop_count: int) -> str:
    """Build prompt for LLM decomposition."""

    relations_str = ', '.join(METAQA_RELATIONS)

    # Build example based on hop count
    if hop_count == 1:
        example = '''Example (1-hop):
Question: "What genre is [The Matrix]?"
{
  "hop_1": {
    "question": "What genre is [The Matrix]?",
    "relation": "has_genre"
  }
}'''
    elif hop_count == 2:
        example = '''Example (2-hop):
Question: "What genre are movies directed by the director of [The Matrix]?"
{
  "hop_1": {
    "question": "Who directed [The Matrix]?",
    "relation": "directed_by"
  },
  "hop_2": {
    "question": "What genre are movies directed by [answer_from_hop_1]?",
    "relation": "has_genre"
  }
}'''
    else:  # 3-hop
        example = '''Example (3-hop):
Question: "When did films written by [Send Me No Flowers] writers release?"
{
  "hop_1": {
    "question": "Who wrote [Send Me No Flowers]?",
    "relation": "written_by"
  },
  "hop_2": {
    "question": "What films were written by [answer_from_hop_1]?",
    "relation": "written_by"
  },
  "hop_3": {
    "question": "When did [answer_from_hop_2] release?",
    "relation": "release_year"
  }
}'''

    prompt = f"""Given a {hop_count}-hop knowledge graph question, decompose it into {hop_count} sequential 1-hop questions.

Available relations: {relations_str}

Question: "{question_text}"

Rules:
1. Each subquestion is a standalone 1-hop question
2. Use [entity_name] for entities from the original question (already in brackets)
3. Use [answer_from_hop_N] as placeholder for answers from previous hops
4. Select ONE relation per hop from the available relations list

{example}

Now decompose the question above. Respond in valid JSON format only."""

    return prompt


def decompose_question_with_llm(
    client: OpenAI,
    question_text: str,
    hop_count: int
) -> Dict:
    """
    Call OpenAI API to decompose question.

    Returns:
        Dict with decomposition or error info
    """
    prompt = build_decomposition_prompt(question_text, hop_count)

    try:
        response = client.chat.completions.create(
            model=Config.OPENAI_MODEL_CHAT,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
            # Note: gpt-5-mini only supports default temperature (1)
        )

        result = json.loads(response.choices[0].message.content)

        # Calculate cost
        usage = response.usage
        input_cost = usage.prompt_tokens * 0.15 / 1_000_000  # gpt-4o-mini pricing
        output_cost = usage.completion_tokens * 0.60 / 1_000_000
        cost = input_cost + output_cost

        return {
            'success': True,
            'decomposition': result,
            'cost': cost,
            'tokens': {
                'prompt': usage.prompt_tokens,
                'completion': usage.completion_tokens,
                'total': usage.total_tokens
            }
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def validate_decomposition(decomposition: Dict, hop_count: int) -> List[str]:
    """
    Validate decomposition format and content.

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check all hops are present
    for hop_num in range(1, hop_count + 1):
        hop_key = f"hop_{hop_num}"
        if hop_key not in decomposition:
            errors.append(f"Missing {hop_key}")
            continue

        hop_data = decomposition[hop_key]

        # Check question field
        if 'question' not in hop_data:
            errors.append(f"{hop_key}: Missing 'question' field")
        elif not isinstance(hop_data['question'], str):
            errors.append(f"{hop_key}: 'question' must be string")

        # Check relation field
        if 'relation' not in hop_data:
            errors.append(f"{hop_key}: Missing 'relation' field")
        elif hop_data['relation'] not in METAQA_RELATIONS:
            errors.append(f"{hop_key}: Invalid relation '{hop_data['relation']}'")

        # Check placeholder usage for hop 2+
        if hop_num > 1:
            question = hop_data.get('question', '')
            expected_placeholder = f"[answer_from_hop_{hop_num - 1}]"
            if expected_placeholder not in question:
                errors.append(f"{hop_key}: Should use placeholder '{expected_placeholder}'")

    return errors


def print_decomposition(question_text: str, hop_count: int, result: Dict):
    """Pretty print decomposition result."""
    print("=" * 80)
    print(f"Question ({hop_count}-hop):")
    print(f"  {question_text}")
    print()

    if not result['success']:
        print(f"ERROR: {result['error']}")
        return

    decomposition = result['decomposition']

    # Validate
    errors = validate_decomposition(decomposition, hop_count)

    if errors:
        print("VALIDATION ERRORS:")
        for error in errors:
            print(f"  - {error}")
        print()

    # Print decomposition
    print("Decomposition:")
    for hop_num in range(1, hop_count + 1):
        hop_key = f"hop_{hop_num}"
        if hop_key in decomposition:
            hop_data = decomposition[hop_key]
            question = hop_data.get('question', 'N/A')
            relation = hop_data.get('relation', 'N/A')
            print(f"  Hop {hop_num}:")
            print(f"    Question: {question}")
            print(f"    Relation: {relation}")

    # Print cost
    print(f"\nCost: ${result['cost']:.6f}")
    print(f"Tokens: {result['tokens']['prompt']} prompt + {result['tokens']['completion']} completion = {result['tokens']['total']} total")

    # Status
    if not errors:
        print("\nStatus:  VALID")
    else:
        print("\nStatus:  VALIDATION ERRORS")


def main():
    print("=" * 80)
    print("Variant 6 - Phase 0: Test LLM Decomposition")
    print("=" * 80)
    print()

    # Initialize OpenAI client
    print("Initializing OpenAI client...")
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    print(f"Model: {Config.OPENAI_MODEL_CHAT}")
    print()

    # Load sample questions
    print("Loading sample questions...")
    samples_per_hop = 5

    questions_1hop = load_qa_dataset(str(Config.QA_1HOP_TRAIN), hop_count=1)[:samples_per_hop]
    questions_2hop = load_qa_dataset(str(Config.QA_2HOP_TRAIN), hop_count=2)[:samples_per_hop]
    questions_3hop = load_qa_dataset(str(Config.QA_3HOP_TRAIN), hop_count=3)[:samples_per_hop]

    all_samples = [
        (questions_1hop, 1),
        (questions_2hop, 2),
        (questions_3hop, 3)
    ]

    print(f"Loaded {samples_per_hop} samples per hop count")
    print()

    # Test decomposition
    all_results = []
    total_cost = 0.0
    valid_count = 0

    for questions, hop_count in all_samples:
        print(f"\n{'#' * 80}")
        print(f"# Testing {hop_count}-HOP Questions")
        print(f"{'#' * 80}\n")

        for i, question in enumerate(questions, 1):
            print(f"\n[Sample {i}/{samples_per_hop}]")

            result = decompose_question_with_llm(
                client,
                question.text,
                hop_count
            )

            print_decomposition(question.text, hop_count, result)

            # Track results
            if result['success']:
                errors = validate_decomposition(result['decomposition'], hop_count)
                is_valid = len(errors) == 0

                all_results.append({
                    'question_id': question.question_id,
                    'question_text': question.text,
                    'hop_count': hop_count,
                    'decomposition': result['decomposition'],
                    'cost': result['cost'],
                    'tokens': result['tokens'],
                    'valid': is_valid,
                    'errors': errors
                })

                total_cost += result['cost']
                if is_valid:
                    valid_count += 1
            else:
                all_results.append({
                    'question_id': question.question_id,
                    'question_text': question.text,
                    'hop_count': hop_count,
                    'success': False,
                    'error': result['error']
                })

            print()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total samples tested: {len(all_results)}")
    print(f"Valid decompositions: {valid_count}/{len(all_results)}")
    print(f"Total cost: ${total_cost:.6f}")
    print(f"Average cost per question: ${total_cost / len(all_results):.6f}")
    print()

    # Save results
    output_path = Path(__file__).parent / "data" / "phase0_test_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to: {output_path}")
    print()

    # Recommendations
    if valid_count == len(all_results):
        print(" All decompositions are valid! Ready to proceed to Phase 1.")
    else:
        print(" Some decompositions have validation errors. Review the output above.")
        print("  Consider adjusting the prompt before proceeding to Phase 1.")

    print()


if __name__ == "__main__":
    main()
