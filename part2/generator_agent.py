"""
Part 2, Task 4: Generator Agent.
Takes question + context chunks → produces answer + confidence + sources_used.
"""

from typing import TypedDict

from langchain_openai import ChatOpenAI


class GeneratorState(TypedDict):
    question: str
    context_chunks: list[str]
    answer: str
    confidence: float
    sources_used: list[int]


GENERATOR_PROMPT = """Ты — точный аналитический ассистент. Отвечай ИСКЛЮЧИТЕЛЬНО на основе предоставленных фрагментов контекста.

Правила:
1. Используй только информацию из контекста.
2. Если данных нет — скажи об этом прямо.
3. Цитируй конкретные факты и цифры.
4. Укажи, какие фрагменты (по номеру) ты использовал.
5. Оцени уверенность от 0 до 1.

Формат ответа (строго JSON):
{{
  "answer": "твой ответ здесь",
  "confidence": 0.0-1.0,
  "sources_used": [0, 1, ...],
  "sufficient": true/false
}}

Фрагменты контекста:
{context}

Вопрос: {question}"""


def generate_node(state: GeneratorState) -> GeneratorState:
    """LangGraph node: generate answer from context."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    numbered_context = "\n\n".join(
        f"[{i}] {chunk}" for i, chunk in enumerate(state["context_chunks"])
    )

    prompt = GENERATOR_PROMPT.format(
        context=numbered_context,
        question=state["question"],
    )

    response = llm.invoke([{"role": "user", "content": prompt}])

    import json as _json
    import re

    raw = response.content
    # Extract JSON from response
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            parsed = _json.loads(match.group())
            answer = parsed.get("answer", raw)
            confidence = float(parsed.get("confidence", 0.5))
            sources_used = parsed.get("sources_used", [])
        except Exception:
            answer = raw
            confidence = 0.5
            sources_used = []
    else:
        answer = raw
        confidence = 0.5
        sources_used = []

    return {
        **state,
        "answer": answer,
        "confidence": confidence,
        "sources_used": sources_used,
    }


def run_generator(question: str, context_chunks: list[str]) -> dict:
    """Standalone runner for testing."""
    state: GeneratorState = {
        "question": question,
        "context_chunks": context_chunks,
        "answer": "",
        "confidence": 0.0,
        "sources_used": [],
    }
    result = generate_node(state)
    return result


if __name__ == "__main__":
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).parent.parent / ".env")

    test_chunks = [
        "Кустанайская кондитерская фабрика была введена в эксплуатацию в декабре 1974 года.",
        "Проектная мощность составляла 24560 тонн кондитерских изделий в год.",
    ]
    result = run_generator("В каком году основана фабрика?", test_chunks)
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Sources used: {result['sources_used']}")
