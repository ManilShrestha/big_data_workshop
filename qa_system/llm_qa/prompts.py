"""Shared prompts for LLM-based QA"""


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
