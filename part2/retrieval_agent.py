"""
Part 2, Task 3: Retrieval Agent.
Tests BM25 (sparse), HyDE, and rerank retrieval techniques.
Alpha tuning for hybrid search. Evaluates relevance, selects best config.
"""

import json
import re
from typing import TypedDict

import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rank_bm25 import BM25Okapi


class RetrievalState(TypedDict):
    query: str
    chunks: list[str]
    retrieved_docs: dict[str, list[str]]
    retrieval_scores: dict[str, float]
    best_config: str
    best_alpha: float
    final_docs: list[str]


# ─── Embedding helper ────────────────────────────────────────────────────────

_embeddings_model = None

def get_embeddings():
    global _embeddings_model
    if _embeddings_model is None:
        _embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    return _embeddings_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_embeddings().embed_documents(texts)


def embed_query(query: str) -> list[float]:
    return get_embeddings().embed_query(query)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-9))


# ─── Retrieval methods ────────────────────────────────────────────────────────

def retrieve_sparse_bm25(query: str, chunks: list[str], top_k: int = 5) -> list[str]:
    """BM25 lexical search."""
    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices if scores[i] > 0]


def retrieve_hyde(query: str, chunks: list[str], top_k: int = 5) -> list[str]:
    """HyDE: generate hypothetical answer, search by its embedding."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    hypo_prompt = f"Напиши короткий гипотетический ответ (2-3 предложения) на вопрос: {query}"
    hypothetical_answer = llm.invoke([{"role": "user", "content": hypo_prompt}]).content

    chunk_embeddings = embed_texts(chunks)
    query_embedding = embed_query(hypothetical_answer)

    scored = [(cosine_similarity(query_embedding, ce), chunk) for ce, chunk in zip(chunk_embeddings, chunks)]
    scored.sort(key=lambda x: -x[0])
    return [chunk for _, chunk in scored[:top_k]]


def retrieve_dense(query: str, chunks: list[str], top_k: int = 20) -> tuple[list[str], list[float]]:
    """Dense vector search, returns (docs, scores)."""
    chunk_embeddings = embed_texts(chunks)
    query_embedding = embed_query(query)
    scored = [(cosine_similarity(query_embedding, ce), chunk) for ce, chunk in zip(chunk_embeddings, chunks)]
    scored.sort(key=lambda x: -x[0])
    docs = [chunk for _, chunk in scored[:top_k]]
    scores = [score for score, _ in scored[:top_k]]
    return docs, scores


def retrieve_rerank(query: str, chunks: list[str], top_k: int = 5) -> list[str]:
    """Two-stage: dense retrieve top-20, then LLM rerank to top-5."""
    candidate_docs, _ = retrieve_dense(query, chunks, top_k=20)
    if not candidate_docs:
        return chunks[:top_k]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    numbered = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(candidate_docs[:15]))
    rerank_prompt = f"""Оцени каждый фрагмент по релевантности к вопросу (1-10).
Вопрос: {query}

Фрагменты:
{numbered}

Верни JSON: {{"rankings": [{{\"idx\": 0, \"score\": 8}}, ...]}} — только для топ-{top_k}"""

    response = llm.invoke([{"role": "user", "content": rerank_prompt}])
    match = re.search(r'\{.*\}', response.content, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            rankings = parsed.get("rankings", [])
            rankings.sort(key=lambda x: -x.get("score", 0))
            return [candidate_docs[r["idx"]] for r in rankings[:top_k] if r["idx"] < len(candidate_docs)]
        except Exception:
            pass
    return candidate_docs[:top_k]


def retrieve_hybrid(query: str, chunks: list[str], alpha: float = 0.5, top_k: int = 5) -> list[str]:
    """Hybrid: alpha * dense + (1-alpha) * BM25."""
    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.lower().split())

    chunk_embeddings = embed_texts(chunks)
    query_embedding = embed_query(query)
    dense_scores = [cosine_similarity(query_embedding, ce) for ce in chunk_embeddings]

    # Normalize scores to [0, 1]
    bm25_arr = np.array(bm25_scores, dtype=float)
    dense_arr = np.array(dense_scores, dtype=float)
    if bm25_arr.max() > 0:
        bm25_arr = bm25_arr / bm25_arr.max()
    if dense_arr.max() > 0:
        dense_arr = dense_arr / dense_arr.max()

    hybrid_scores = alpha * dense_arr + (1 - alpha) * bm25_arr
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]


# ─── LangGraph nodes ─────────────────────────────────────────────────────────

def try_retrieval_methods(state: RetrievalState) -> RetrievalState:
    """Node 1: run all retrieval methods + alpha tuning."""
    query = state["query"]
    chunks = state["chunks"]
    retrieved_docs = {}

    print(f"  Sparse BM25...")
    retrieved_docs["sparse"] = retrieve_sparse_bm25(query, chunks) or chunks[:5]

    print(f"  HyDE...")
    retrieved_docs["hyde"] = retrieve_hyde(query, chunks)

    print(f"  Rerank (dense top-20 → LLM top-5)...")
    retrieved_docs["rerank"] = retrieve_rerank(query, chunks)

    # Alpha tuning
    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
        key = f"hybrid_alpha={alpha}"
        print(f"  Hybrid alpha={alpha}...")
        retrieved_docs[key] = retrieve_hybrid(query, chunks, alpha=alpha)

    return {**state, "retrieved_docs": retrieved_docs}


def evaluate_retrieval(state: RetrievalState) -> RetrievalState:
    """Node 2: LLM evaluates relevance of retrieved docs (1-10)."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retrieval_scores = {}

    eval_prompt = """Оцени релевантность фрагментов к вопросу от 1 до 10.
Вопрос: {query}
Фрагменты:
{docs}

Ответь JSON: {{"score": 0-10, "reason": "..."}}"""

    for method, docs in state["retrieved_docs"].items():
        if not docs:
            retrieval_scores[method] = 0.0
            continue
        numbered = "\n\n".join(f"[{i}] {d}" for i, d in enumerate(docs[:5]))
        response = llm.invoke([{
            "role": "user",
            "content": eval_prompt.format(query=state["query"], docs=numbered)
        }])
        match = re.search(r'\{.*\}', response.content, re.DOTALL)
        score = 5.0
        if match:
            try:
                score = float(json.loads(match.group()).get("score", 5.0))
            except Exception:
                pass
        retrieval_scores[method] = score
        print(f"  [{method}] relevance_score={score}")

    return {**state, "retrieval_scores": retrieval_scores}


def select_best_retrieval(state: RetrievalState) -> RetrievalState:
    """Node 3: select best method + best alpha."""
    best_method = max(state["retrieval_scores"], key=lambda k: state["retrieval_scores"][k])
    best_score = state["retrieval_scores"][best_method]

    # Extract best alpha from hybrid configs
    hybrid_scores = {
        k: v for k, v in state["retrieval_scores"].items()
        if k.startswith("hybrid_alpha")
    }
    if hybrid_scores:
        best_hybrid = max(hybrid_scores, key=lambda k: hybrid_scores[k])
        best_alpha = float(best_hybrid.split("=")[1])
    else:
        best_alpha = 0.5

    print(f"\n  Best retrieval: '{best_method}' (score={best_score})")
    print(f"  Best alpha: {best_alpha}")

    return {
        **state,
        "best_config": best_method,
        "best_alpha": best_alpha,
        "final_docs": state["retrieved_docs"][best_method],
    }


# ─── Public API ──────────────────────────────────────────────────────────────

def build_retrieval_graph():
    from langgraph.graph import END, StateGraph

    graph = StateGraph(RetrievalState)
    graph.add_node("try_retrieval_methods", try_retrieval_methods)
    graph.add_node("evaluate_retrieval", evaluate_retrieval)
    graph.add_node("select_best_retrieval", select_best_retrieval)

    graph.set_entry_point("try_retrieval_methods")
    graph.add_edge("try_retrieval_methods", "evaluate_retrieval")
    graph.add_edge("evaluate_retrieval", "select_best_retrieval")
    graph.add_edge("select_best_retrieval", END)

    return graph.compile()


def run_retrieval_agent(query: str, chunks: list[str]) -> RetrievalState:
    app = build_retrieval_graph()
    initial: RetrievalState = {
        "query": query,
        "chunks": chunks,
        "retrieved_docs": {},
        "retrieval_scores": {},
        "best_config": "",
        "best_alpha": 0.5,
        "final_docs": [],
    }
    return app.invoke(initial)


if __name__ == "__main__":
    from dotenv import load_dotenv
    from pathlib import Path
    import pdfplumber

    load_dotenv(Path(__file__).parent.parent / ".env")

    pdf_path = Path(__file__).parent.parent / "data" / "bsulp_2024_rus_pages_1_15.pdf"
    with pdfplumber.open(str(pdf_path)) as pdf:
        texts = [p.extract_text() or "" for p in pdf.pages]

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))

    result = run_retrieval_agent("В каком году основана фабрика?", chunks)
    print(f"\nBest config: {result['best_config']}")
    print(f"Best alpha: {result['best_alpha']}")
    print(f"Final docs count: {len(result['final_docs'])}")

    print("\n=== ALPHA TUNING LOG ===")
    for k, v in sorted(result["retrieval_scores"].items()):
        print(f"  {k:<30} | score={v:.1f}")
