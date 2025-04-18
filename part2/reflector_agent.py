"""
Part 2, Task 5: Reflector Agent.
Evaluates answer quality on 3 criteria, routes to correct retry target.
"""

import json
import re
from typing import Literal, TypedDict

from langchain_openai import ChatOpenAI


class ReflectorState(TypedDict):
    question: str
    context_chunks: list[str]
    answer: str
    confidence: float
    attempt: int
    scores: dict[str, float]
    decision: str
    feedback: str
    retry_target: str


REFLECT_PROMPT = """Ты — строгий оценщик качества ответов RAG-системы.

Оцени ответ по 3 критериям от 1 до 10:
1. **Relevance** (Релевантность): Насколько контекст отвечает на вопрос?
2. **Faithfulness** (Верность): Опирается ли ответ ТОЛЬКО на контекст (нет выдумок)?
3. **Completeness** (Полнота): Полностью ли отвечает на вопрос?

Вопрос: {question}

Контекст (фрагменты):
{context}

Ответ: {answer}

Верни строго JSON:
{{
  "relevance": 1-10,
  "faithfulness": 1-10,
  "completeness": 1-10,
  "feedback": "что не так и что улучшить",
  "has_garbage": false
}}"""


def reflect_node(state: ReflectorState) -> ReflectorState:
    """LangGraph node: evaluate answer, decide next action."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    attempt = state.get("attempt", 0) + 1

    context_str = "\n\n".join(
        f"[{i}] {c}" for i, c in enumerate(state["context_chunks"])
    )

    response = llm.invoke([{
        "role": "user",
        "content": REFLECT_PROMPT.format(
            question=state["question"],
            context=context_str[:3000],
            answer=state["answer"],
        ),
    }])

    raw = response.content
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
        except Exception:
            parsed = {}
    else:
        parsed = {}

    relevance = float(parsed.get("relevance", 5))
    faithfulness = float(parsed.get("faithfulness", 5))
    completeness = float(parsed.get("completeness", 5))
    has_garbage = bool(parsed.get("has_garbage", False))
    feedback = parsed.get("feedback", "No feedback")

    scores = {
        "relevance": relevance,
        "faithfulness": faithfulness,
        "completeness": completeness,
    }

    print(f"  [Reflector attempt={attempt}] relevance={relevance} faithfulness={faithfulness} completeness={completeness}")

    # Routing logic
    if attempt >= 3:
        decision = "accept"
        retry_target = "end"
        print(f"  Force accept (max attempts reached)")
    elif has_garbage or (relevance < 5 and faithfulness < 5):
        decision = "retry_extraction"
        retry_target = "extract"
        print(f"  → retry_extraction (garbage or very low scores)")
    elif relevance < 7:
        decision = "retry_retrieval"
        retry_target = "retrieve"
        print(f"  → retry_retrieval (relevance={relevance} < 7)")
    elif faithfulness < 7:
        decision = "retry_generation"
        retry_target = "generate"
        print(f"  → retry_generation (faithfulness={faithfulness} < 7)")
    else:
        decision = "accept"
        retry_target = "end"
        print(f"  → accept (all scores >= 7)")

    return {
        **state,
        "attempt": attempt,
        "scores": scores,
        "decision": decision,
        "feedback": feedback,
        "retry_target": retry_target,
    }


def route_after_reflect(state: ReflectorState) -> str:
    """Conditional edge: return next node name."""
    return state["retry_target"]


def build_reflector_graph():
    """Standalone reflector graph (for testing)."""
    from langgraph.graph import END, StateGraph

    graph = StateGraph(ReflectorState)
    graph.add_node("reflect", reflect_node)
    graph.set_entry_point("reflect")
    graph.add_edge("reflect", END)
    return graph.compile()


def run_reflector(
    question: str,
    context_chunks: list[str],
    answer: str,
    confidence: float = 0.5,
    attempt: int = 0,
) -> ReflectorState:
    state: ReflectorState = {
        "question": question,
        "context_chunks": context_chunks,
        "answer": answer,
        "confidence": confidence,
        "attempt": attempt,
        "scores": {},
        "decision": "",
        "feedback": "",
        "retry_target": "",
    }
    return reflect_node(state)


if __name__ == "__main__":
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).parent.parent / ".env")

    result = run_reflector(
        question="В каком году основана фабрика?",
        context_chunks=["Фабрика введена в 1974 году."],
        answer="Фабрика была основана в 1974 году.",
        attempt=0,
    )
    print(f"Scores: {result['scores']}")
    print(f"Decision: {result['decision']}")
    print(f"Feedback: {result['feedback']}")
