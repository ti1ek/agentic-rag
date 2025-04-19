"""
Part 2, Task 6: Main Orchestrator — LangGraph with all 5 agents.
Extraction Agent called via A2A (HTTP). Others are inline nodes.
"""

import json
from pathlib import Path
from typing import TypedDict

import httpx
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(Path(__file__).parent.parent / ".env")

from chunking_agent import run_chunking_agent
from generator_agent import generate_node, GeneratorState
from reflector_agent import reflect_node, ReflectorState
from retrieval_agent import run_retrieval_agent

ROOT = Path(__file__).parent.parent
PDF_PATH = str(ROOT / "data" / "bsulp_2024_rus_pages_1_15.pdf")
EXTRACTION_AGENT_URL = "http://localhost:5010"


class OrchestratorState(TypedDict):
    question: str
    pdf_path: str
    # Extraction
    final_text: str
    best_extraction_method: str
    vl_scores: dict
    # Chunking
    final_chunks: list[str]
    best_chunking_strategy: str
    # Retrieval
    final_docs: list[str]
    best_retrieval_config: str
    best_alpha: float
    # Generation
    answer: str
    confidence: float
    sources_used: list[int]
    # Reflection
    attempt: int
    scores: dict
    decision: str
    feedback: str
    retry_target: str


# ─── Nodes ────────────────────────────────────────────────────────────────────

def extract_node(state: OrchestratorState) -> OrchestratorState:
    """Call Extraction Agent via A2A HTTP."""
    print(f"\n[EXTRACT] Calling Extraction Agent at {EXTRACTION_AGENT_URL}...")
    try:
        with httpx.Client(timeout=300.0) as client:
            response = client.post(
                f"{EXTRACTION_AGENT_URL}/a2a",
                json={
                    "skill": "extract_pdf",
                    "input": {
                        "pdf_path": state["pdf_path"],
                        "pages": list(range(15)),
                        "methods": ["pdfplumber", "vlm", "docling"],
                    },
                },
            )
            data = response.json()

        if data.get("error"):
            raise RuntimeError(f"A2A error: {data['error']}")

        output = data["output"]
        print(f"  Best extraction method: {output['best_method']}")
        return {
            **state,
            "final_text": output["final_text"],
            "best_extraction_method": output["best_method"],
            "vl_scores": output.get("vl_scores", {}),
        }
    except Exception as e:
        print(f"  [EXTRACT] A2A call failed ({e}), falling back to pdfplumber")
        import pdfplumber
        with pdfplumber.open(state["pdf_path"]) as pdf:
            texts = [p.extract_text() or "" for p in pdf.pages]
        return {
            **state,
            "final_text": "\n\n".join(texts),
            "best_extraction_method": "pdfplumber_fallback",
            "vl_scores": {},
        }


def chunk_node(state: OrchestratorState) -> OrchestratorState:
    """Run Chunking Agent."""
    print(f"\n[CHUNK] Running Chunking Agent...")
    feedback = state.get("feedback", "")

    # If reflector suggested retry due to garbage text, try simpler chunking
    texts = [state["final_text"]]
    result = run_chunking_agent(texts)

    print(f"  Best strategy: {result['best_strategy']} ({len(result['final_chunks'])} chunks)")
    return {
        **state,
        "final_chunks": result["final_chunks"],
        "best_chunking_strategy": result["best_strategy"],
    }


def retrieve_node(state: OrchestratorState) -> OrchestratorState:
    """Run Retrieval Agent."""
    print(f"\n[RETRIEVE] Running Retrieval Agent...")
    result = run_retrieval_agent(state["question"], state["final_chunks"])

    print(f"  Best config: {result['best_config']}, alpha={result['best_alpha']}")
    return {
        **state,
        "final_docs": result["final_docs"],
        "best_retrieval_config": result["best_config"],
        "best_alpha": result["best_alpha"],
    }


def generate_node_wrapper(state: OrchestratorState) -> OrchestratorState:
    """Run Generator Agent."""
    print(f"\n[GENERATE] Running Generator Agent...")
    gen_state: GeneratorState = {
        "question": state["question"],
        "context_chunks": state["final_docs"],
        "answer": "",
        "confidence": 0.0,
        "sources_used": [],
    }
    result = generate_node(gen_state)
    print(f"  Answer: {result['answer'][:100]}...")
    print(f"  Confidence: {result['confidence']}")
    return {
        **state,
        "answer": result["answer"],
        "confidence": result["confidence"],
        "sources_used": result["sources_used"],
    }


def reflect_node_wrapper(state: OrchestratorState) -> OrchestratorState:
    """Run Reflector Agent."""
    print(f"\n[REFLECT] Running Reflector Agent (attempt={state.get('attempt', 0)+1})...")
    ref_state: ReflectorState = {
        "question": state["question"],
        "context_chunks": state["final_docs"],
        "answer": state["answer"],
        "confidence": state["confidence"],
        "attempt": state.get("attempt", 0),
        "scores": {},
        "decision": "",
        "feedback": "",
        "retry_target": "",
    }
    result = reflect_node(ref_state)
    return {
        **state,
        "attempt": result["attempt"],
        "scores": result["scores"],
        "decision": result["decision"],
        "feedback": result["feedback"],
        "retry_target": result["retry_target"],
    }


def route_after_reflect(state: OrchestratorState) -> str:
    """Conditional edge from reflect node."""
    target = state.get("retry_target", "end")
    print(f"  [ROUTE] → {target}")
    return target


# ─── Build graph ──────────────────────────────────────────────────────────────

def build_orchestrator():
    from langgraph.graph import END, StateGraph

    graph = StateGraph(OrchestratorState)
    graph.add_node("extract", extract_node)
    graph.add_node("chunk", chunk_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node_wrapper)
    graph.add_node("reflect", reflect_node_wrapper)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "chunk")
    graph.add_edge("chunk", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "reflect")
    graph.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {
            "end": END,
            "retrieve": "retrieve",
            "generate": "generate",
            "extract": "extract",
        },
    )

    return graph.compile()


def save_graph_visualization(app, output_path: str = "orchestrator_graph.png"):
    """Save LangGraph visualization as PNG."""
    try:
        png = app.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(png)
        print(f"Graph visualization saved to {output_path}")
    except Exception as e:
        print(f"Could not save graph visualization: {e}")

        # Save mermaid text as fallback
        try:
            mermaid = app.get_graph().draw_mermaid()
            txt_path = output_path.replace(".png", ".mmd")
            with open(txt_path, "w") as f:
                f.write(mermaid)
            print(f"Mermaid diagram saved to {txt_path}")
        except Exception:
            pass


def run_pipeline(question: str, pdf_path: str = PDF_PATH) -> OrchestratorState:
    """Run the full multi-agent pipeline for a question."""
    app = build_orchestrator()

    initial: OrchestratorState = {
        "question": question,
        "pdf_path": pdf_path,
        "final_text": "",
        "best_extraction_method": "",
        "vl_scores": {},
        "final_chunks": [],
        "best_chunking_strategy": "",
        "final_docs": [],
        "best_retrieval_config": "",
        "best_alpha": 0.5,
        "answer": "",
        "confidence": 0.0,
        "sources_used": [],
        "attempt": 0,
        "scores": {},
        "decision": "",
        "feedback": "",
        "retry_target": "",
    }

    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"{'='*60}")

    result = app.invoke(initial)

    print(f"\n{'='*60}")
    print(f"FINAL ANSWER: {result['answer']}")
    print(f"CONFIDENCE: {result['confidence']}")
    print(f"EXTRACTION: {result['best_extraction_method']}")
    print(f"CHUNKING: {result['best_chunking_strategy']}")
    print(f"RETRIEVAL: {result['best_retrieval_config']} (alpha={result['best_alpha']})")
    print(f"SCORES: {result['scores']}")
    print(f"ATTEMPTS: {result['attempt']}")
    print(f"{'='*60}")

    return result


if __name__ == "__main__":
    app = build_orchestrator()
    save_graph_visualization(app, str(ROOT / "part2" / "orchestrator_graph.png"))

    result = run_pipeline("В каком году была введена в эксплуатацию Кустанайская кондитерская фабрика?")
    print(json.dumps({
        "answer": result["answer"],
        "confidence": result["confidence"],
        "scores": result["scores"],
        "extraction_method": result["best_extraction_method"],
    }, ensure_ascii=False, indent=2))
