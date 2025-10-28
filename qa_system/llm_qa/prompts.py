"""Shared prompts for LLM-based QA"""


def build_qwen_qa_prompt(question: str) -> str:
    """
    Build a simpler, more constrained prompt for Qwen models (single-pass)

    Key differences from GPT-4o prompt:
    1. Simpler instructions (avoid confusion)
    2. Stronger JSON format constraints
    3. Explicit anti-hallucination warning
    4. Fewer examples (avoid pattern copying)
    5. No placeholder examples like "item1", "item2"
    6. Generic enough for all answer types (movies, people, genres, years, etc.)
    7. Emphasizes listing ALL known answers (not just 2-3)

    Args:
        question: Question text

    Returns:
        Formatted prompt string
    """
    return f"""Answer this movie database question. Read the question carefully and provide the correct type of answer.

Question: {question}

CRITICAL RULES:
1. Return ONLY valid JSON: {{"answers": ["answer1", "answer2", ...]}}
2. Read what the question is asking for:
   - If it asks for movies → list ALL movie titles you know
   - If it asks for people/actors/directors → list ALL person names you know
   - If it asks for genres → list ALL genre names you know (Action, Drama, Comedy, etc.)
   - If it asks for years → list ALL years you know (1999, 2000, etc.)
   - If it asks for languages → list ALL languages you know (English, French, etc.)
3. List as many answers as you know (aim for 10-20 if available)
4. Include ONLY real factual answers - NO made-up information
5. If you don't know ANY answers, return: {{"answers": []}}
6. NO explanations, NO extra text, ONLY the JSON object
7. Do NOT repeat the same answer multiple times

Your JSON response:"""


def build_qwen_qa_prompt_pass1(question: str) -> str:
    """
    Build the prompt for Qwen Pass 1: Get raw answers (no JSON required)

    Two-pass approach is more reliable for weaker models:
    - Pass 1: Answer the question in plain text (easier)
    - Pass 2: Convert to JSON (separate concern)

    Args:
        question: Question text

    Returns:
        Formatted prompt string
    """
    return f"""Answer this movie database question. Read carefully to provide the correct type of answer.

Question: {question}

Instructions:
- Read what the question asks for (movies, people, genres, years, languages, etc.)
- List ALL the answers you know - don't stop at just 2 or 3
- Include as many factual answers as you can (aim for as many as possible and you are confident about)
- One answer per line
- Only include real information - no made-up answers
- If you don't know ANY answers, say "I don't know"

Your answer (list ALL you know, one per line):"""


def build_qwen_formatting_prompt_pass2(question: str, raw_answer: str) -> str:
    """
    Build the prompt for Qwen Pass 2: Convert raw answer to JSON

    This separates JSON formatting from content generation,
    reducing errors for models that struggle with structured output.

    Args:
        question: Original question text
        raw_answer: Raw answer from Pass 1

    Returns:
        Formatted prompt string
    """
    return f"""Convert this answer into JSON format.

Question: {question}
Answer: {raw_answer}

Convert the answer above into this exact JSON format:
{{"answers": ["item1", "item2", "item3"]}}

Rules:
- Each line becomes one item in the array
- Skip lines like "I don't know" or empty lines
- Keep the exact text, just put it in JSON format
- If there are no valid answers, return: {{"answers": []}}

JSON output:"""


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
