"""Shared prompts for LLM-based QA"""


def build_qwen_qa_prompt(question: str) -> str:
    """
    Build a simpler, more constrained prompt for Qwen models

    Key differences from GPT-4o prompt:
    1. Simpler instructions (avoid confusion)
    2. Stronger JSON format constraints
    3. Explicit anti-hallucination warning
    4. Fewer examples (avoid pattern copying)
    5. No placeholder examples like "item1", "item2"

    Args:
        question: Question text

    Returns:
        Formatted prompt string
    """
    return f"""You are answering a movie database question. Provide ONLY actual movies/people you are confident about.

Question: {question}

STRICT RULES:
1. Return ONLY valid JSON in this exact format: {{"answers": ["Movie 1", "Movie 2"]}}
2. Include ONLY real movies/people - NO made-up titles
3. If you don't know, return: {{"answers": []}}
4. NO explanations, NO extra text, ONLY the JSON object
5. Movie titles WITHOUT years: "The Matrix" not "The Matrix (1999)"
6. Do NOT repeat the same movie multiple times

Example output format:
{{"answers": ["Forrest Gump", "Cast Away", "Apollo 13"]}}

Your JSON response:"""


def build_direct_qa_prompt(question: str) -> str:
    """
    Build the prompt for direct QA (used by both DirectLLMQA and BatchLLMQA)

    Args:
        question: Question text

    Returns:
        Formatted prompt string
    """
    return f"""Answer this movie question with a comprehensive list of specific factual answers.

Question: {question}

CRITICAL REQUIREMENTS:
- You MUST provide direct answers in JSON format - do NOT ask for clarification
- List up to 20 relevant answers if available
- Include ALL major and minor films/people you know
- Return movie titles WITHOUT years: "The Matrix" not "The Matrix (1999)"
- Format as JSON: {{"answers": ["item1", "item2", ...]}}
- NO explanations, questions, or extra text - ONLY the JSON object

Examples:
Q: "Who directed [The Matrix]?"
A: {{"answers": ["Lana Wachowski", "Lilly Wachowski"]}}

Q: "What movies did [Tom Hanks] star in?"
A: {{"answers": ["Forrest Gump", "Saving Private Ryan", "Cast Away", "The Green Mile", "Toy Story", "Philadelphia", "Big", "Sleepless in Seattle", "Apollo 13", "The Terminal"]}}

Q: "[Helen Mack] appears in which movies"
A: {{"answers": ["Son of Kong", "She", "The Milky Way", "Kiss and Make-Up", "College Humor"]}}

Q: "what films does [Paresh Rawal] appear in"
A: {{"answers": ["Hera Pheri", "Welcome", "Phir Hera Pheri", "Sardar", "Andaz Apna Apna", "Naam", "Mohra", "Oh My God", "Babu Rao Apte"]}}

Provide your answer now as JSON with ALL movies/people you know:"""
