#!/usr/bin/env python3
"""
Test inference with fine-tuned Qwen model
"""

import json
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


METAQA_RELATIONS = [
    'directed_by',
    'starred_actors',
    'written_by',
    'in_language',
    'has_genre',
    'release_year',
    'has_tags',
    'has_imdb_rating',
    'has_imdb_votes',
]


def build_test_prompt(question: str, hop_count: int) -> list:
    """Build prompt in the same format as training data"""
    relations_list = "\n".join([f"- {rel}" for rel in METAQA_RELATIONS])

    system_prompt = """You are analyzing a knowledge graph question to determine which relations to traverse at each hop.

Your task is to select which relation(s) are most relevant at EACH hop to answer the question.

Think step-by-step about the logical path needed to answer the question."""

    user_prompt = f"""Available relations in the knowledge graph:
{relations_list}

Question: "{question}"

This is a {hop_count}-hop question. You need to select which relation(s) might be relevant at EACH hop.

For each hop, select 1-2 most relevant relations from the list above. Think about the logical path needed to answer the question.

Respond in JSON format:
{{
  "reasoning": "brief explanation of the path",
  "hops": [
    ["relation1"],
    ["relation2"],
    ...
  ]
}}

Provide your analysis:"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def main():
    parser = argparse.ArgumentParser(description='Test fine-tuned Qwen model')
    parser.add_argument(
        '--model',
        type=str,
        default='models/qwen2.5-7b-lora-variant7',
        help='Path to LoRA model checkpoint'
    )
    parser.add_argument(
        '--question',
        type=str,
        default='who directed [The Matrix]',
        help='Test question'
    )
    parser.add_argument(
        '--hop-count',
        type=int,
        default=1,
        help='Number of hops in question'
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print(" Testing Fine-Tuned Qwen Model")
    print("="*80)
    print(f"  Model: {args.model}")
    print(f"  Question: {args.question}")
    print(f"  Hop count: {args.hop_count}")
    print("="*80 + "\n")

    # Load model
    print("Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        device_map="auto",
        torch_dtype="auto"
    )

    model = PeftModel.from_pretrained(base_model, args.model)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    print("   Model loaded\n")

    # Build prompt
    messages = build_test_prompt(args.question, args.hop_count)

    # Generate
    print("Generating response...")
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    # Decode
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

    print("\n" + "="*80)
    print(" Model Response")
    print("="*80)
    print(response)
    print("="*80 + "\n")

    # Try to parse JSON
    try:
        parsed = json.loads(response)
        print(" Valid JSON output!")
        print(f"\nReasoning: {parsed.get('reasoning', 'N/A')}")
        print(f"Hops: {parsed.get('hops', [])}")
    except json.JSONDecodeError:
        print(" Warning: Could not parse as JSON")


if __name__ == "__main__":
    main()
