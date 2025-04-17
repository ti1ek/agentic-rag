"""
Part 2, Task 2: Chunking Agent.
Experiments with 5 chunking strategies, evaluates quality, selects best.
"""

import json
import random
import re
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChunkingState(TypedDict):
    texts: list[str]
    chunks_by_strategy: dict[str, list[str]]
    quality_metrics: dict[str, dict]
    best_strategy: str
    final_chunks: list[str]


# ─── Chunking strategies ────────────────────────────────────────────────────

def chunk_recursive(texts: list[str]) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return [c for c in chunks if len(c.strip()) > 20]


def chunk_small(texts: list[str]) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return [c for c in chunks if len(c.strip()) > 10]


def chunk_large(texts: list[str]) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return [c for c in chunks if len(c.strip()) > 30]


def chunk_by_section(texts: list[str]) -> list[str]:
    """Split by section headings (numbered sections, CAPS lines)."""
    heading_pattern = re.compile(
        r'(?=^(?:\d+\.\d*\s+[А-ЯA-Z]|\d+\s+[А-ЯA-Z]{3,}|[А-ЯA-Z]{4,}\s))',
        re.MULTILINE,
    )
    chunks = []
    for text in texts:
        parts = heading_pattern.split(text)
        for part in parts:
            part = part.strip()
            if len(part) > 50:
                chunks.append(part)
    if not chunks:
        chunks = chunk_recursive(texts)
    return chunks


def chunk_semantic(texts: list[str]) -> list[str]:
    """Split by sentence boundaries (simple semantic approximation)."""
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[А-ЯA-Z])')
    all_sentences = []
    for text in texts:
        sentences = sentence_endings.split(text)
        all_sentences.extend([s.strip() for s in sentences if len(s.strip()) > 20])

    # Group sentences into ~400-char chunks
    chunks = []
    current = []
    current_len = 0
    for sent in all_sentences:
        if current_len + len(sent) > 400 and current:
            chunks.append(" ".join(current))
            current = [sent]
            current_len = len(sent)
        else:
            current.append(sent)
            current_len += len(sent)
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c) > 30]


STRATEGIES = {
    "recursive": chunk_recursive,
    "small_chunks": chunk_small,
    "large_chunks": chunk_large,
    "by_section": chunk_by_section,
    "semantic": chunk_semantic,
}


# ─── LangGraph nodes ─────────────────────────────────────────────────────────

def chunk_with_strategies(state: ChunkingState) -> ChunkingState:
    """Node 1: produce chunks for each strategy."""
    chunks_by_strategy = {}
    for name, fn in STRATEGIES.items():
        chunks = fn(state["texts"])
        chunks_by_strategy[name] = chunks
        print(f"  Strategy '{name}': {len(chunks)} chunks, avg_len={sum(len(c) for c in chunks)//max(len(chunks),1)}")
    return {**state, "chunks_by_strategy": chunks_by_strategy}


def evaluate_chunks(state: ChunkingState) -> ChunkingState:
    """Node 2: LLM evaluates 5 random chunks per strategy."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    quality_metrics = {}

    eval_prompt = """Оцени качество следующих фрагментов текста из 1 до 10 по критериям:
- Семантическая связность (нет обрывов посередине мысли)
- Информативность (содержит полезную информацию)
- Отсутствие артефактов (нет обрезанных слов, мусора)

Фрагменты:
{chunks}

Ответь ТОЛЬКО JSON:
{{"avg_score": 0.0-10.0, "semantic_coherence": 0-10, "informativeness": 0-10, "no_artifacts": 0-10, "comment": "..."}}"""

    for strategy_name, chunks in state["chunks_by_strategy"].items():
        sample = random.sample(chunks, min(5, len(chunks)))
        numbered = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(sample))

        response = llm.invoke([{"role": "user", "content": eval_prompt.format(chunks=numbered)}])
        raw = response.content

        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                metrics = json.loads(match.group())
            except Exception:
                metrics = {"avg_score": 5.0}
        else:
            metrics = {"avg_score": 5.0}

        metrics["count"] = len(chunks)
        metrics["avg_len"] = sum(len(c) for c in chunks) // max(len(chunks), 1)
        quality_metrics[strategy_name] = metrics
        print(f"  [{strategy_name}] avg_score={metrics.get('avg_score', '?')}, count={len(chunks)}")

    return {**state, "quality_metrics": quality_metrics}


def select_best_strategy(state: ChunkingState) -> ChunkingState:
    """Node 3: pick strategy with highest avg_score."""
    best = max(
        state["quality_metrics"].items(),
        key=lambda kv: float(kv[1].get("avg_score", 0)),
    )
    best_name = best[0]
    print(f"\n  Best chunking strategy: '{best_name}' (score={best[1].get('avg_score')})")
    return {
        **state,
        "best_strategy": best_name,
        "final_chunks": state["chunks_by_strategy"][best_name],
    }


# ─── Public API ──────────────────────────────────────────────────────────────

def build_chunking_graph():
    """Build and return the chunking LangGraph."""
    from langgraph.graph import END, StateGraph

    graph = StateGraph(ChunkingState)
    graph.add_node("chunk_with_strategies", chunk_with_strategies)
    graph.add_node("evaluate_chunks", evaluate_chunks)
    graph.add_node("select_best_strategy", select_best_strategy)

    graph.set_entry_point("chunk_with_strategies")
    graph.add_edge("chunk_with_strategies", "evaluate_chunks")
    graph.add_edge("evaluate_chunks", "select_best_strategy")
    graph.add_edge("select_best_strategy", END)

    return graph.compile()


def run_chunking_agent(texts: list[str]) -> ChunkingState:
    """Run the chunking agent and return final state."""
    app = build_chunking_graph()
    initial: ChunkingState = {
        "texts": texts,
        "chunks_by_strategy": {},
        "quality_metrics": {},
        "best_strategy": "",
        "final_chunks": [],
    }
    return app.invoke(initial)


def print_metrics_table(quality_metrics: dict):
    """Print a comparison table of chunking strategies."""
    print("\n=== CHUNKING STRATEGIES COMPARISON ===")
    print(f"{'Strategy':<15} | {'Score':>5} | {'Count':>5} | {'AvgLen':>6} | {'Coherence':>9} | {'Informativeness':>15} | {'No Artifacts':>12}")
    print("-" * 90)
    for name, m in sorted(quality_metrics.items(), key=lambda kv: -float(kv[1].get("avg_score", 0))):
        print(
            f"{name:<15} | {float(m.get('avg_score',0)):>5.1f} | {m.get('count',0):>5} | "
            f"{m.get('avg_len',0):>6} | {m.get('semantic_coherence','?'):>9} | "
            f"{m.get('informativeness','?'):>15} | {m.get('no_artifacts','?'):>12}"
        )


if __name__ == "__main__":
    from dotenv import load_dotenv
    from pathlib import Path
    import pdfplumber

    load_dotenv(Path(__file__).parent.parent / ".env")

    pdf_path = Path(__file__).parent.parent / "data" / "bsulp_2024_rus_pages_1_15.pdf"
    with pdfplumber.open(str(pdf_path)) as pdf:
        texts = [p.extract_text() or "" for p in pdf.pages]

    print("Running Chunking Agent...")
    result = run_chunking_agent(texts)
    print_metrics_table(result["quality_metrics"])
    print(f"\nBest strategy: {result['best_strategy']}")
    print(f"Final chunks count: {len(result['final_chunks'])}")
