#!/usr/bin/env python3
"""
Test the fine-tuned model on multiple questions
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import json

# Paths
BASE_MODEL = "/storage/ms5267@drexel.edu/models/qwen3-4b-instruct-2507"
LORA_MODEL = "models/qwen3_4b_test_tiny"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA weights
model = PeftModel.from_pretrained(model, LORA_MODEL)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print(" Model loaded!\n")

# Test questions
test_cases = [
    {
        "question": "who directed [The Matrix]",
        "hop_count": 1,
        "expected": "directed_by"
    },
    {
        "question": "what genre are films by the director of [Titanic]",
        "hop_count": 2,
        "expected": ["directed_by", "has_genre"]
    },
    {
        "question": "what year were movies made by directors of films starring [Tom Cruise] released",
        "hop_count": 3,
        "expected": ["starred_actors", "directed_by", "release_year"]
    }
]

relations = """- directed_by
- starred_actors
- written_by
- in_language
- has_genre
- release_year
- has_tags
- has_imdb_rating
- has_imdb_votes"""

system_msg = """You are analyzing a knowledge graph question to determine which relations to traverse at each hop.

Your task is to select which relation(s) are most relevant at EACH hop to answer the question.

Think step-by-step about the logical path needed to answer the question."""

def test_question(question, hop_count):
    """Test a single question"""

    user_msg = f"""Available relations in the knowledge graph:
{relations}

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

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    # Generate
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True
        )

    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

    return response

# Run tests
print("="*80)
print("TESTING FINE-TUNED MODEL")
print("="*80)

for i, test_case in enumerate(test_cases, 1):
    print(f"\n{'='*80}")
    print(f"TEST {i}: {test_case['hop_count']}-HOP QUESTION")
    print(f"{'='*80}")
    print(f"Question: {test_case['question']}")
    print(f"Expected: {test_case['expected']}")
    print(f"\nGenerating response...\n")

    response = test_question(test_case['question'], test_case['hop_count'])

    print("Model Response:")
    print("-"*80)
    print(response)
    print("-"*80)

    # Try to parse JSON
    try:
        parsed = json.loads(response)
        print("\n Valid JSON!")
        print(f"Reasoning: {parsed.get('reasoning')}")
        print(f"Predicted hops: {parsed.get('hops')}")

        # Check if correct
        predicted_hops = parsed.get('hops', [])
        if len(predicted_hops) == test_case['hop_count']:
            print(f" Correct number of hops: {len(predicted_hops)}")
        else:
            print(f" Wrong number of hops: {len(predicted_hops)} (expected {test_case['hop_count']})")

    except json.JSONDecodeError as e:
        print(f"\n Invalid JSON: {e}")
        print("Model needs more training or different prompt")

print(f"\n{'='*80}")
print("TESTING COMPLETE")
print(f"{'='*80}\n")
