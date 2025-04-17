"""
Part 2, Task 1: Extraction Agent — internal LangGraph.
5 nodes: prepare_images → run_parser → vl_reflect → pick_next_method → select_best.
"""

import base64
import io
import json
import re
from pathlib import Path
from typing import TypedDict

import pdfplumber
from langchain_openai import ChatOpenAI


class ExtractionState(TypedDict):
    pdf_path: str
    pages: list[int]
    methods: list[str]
    method_queue: list[str]
    current_method: str
    raw_extractions: dict[str, str]
    vl_scores: dict[str, dict]
    page_images: list[str]  # base64 PNGs
    attempt: int
    final_text: str
    best_method: str
    methods_tried: list[str]


DEFAULT_METHODS = ["pdfplumber", "vlm", "docling"]


# ─── Extraction methods ───────────────────────────────────────────────────────

def extract_pdfplumber(pdf_path: str, pages: list[int]) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        texts = []
        for i in pages:
            if i < len(pdf.pages):
                text = pdf.pages[i].extract_text() or ""
                texts.append(text)
    return "\n\n".join(texts)


def extract_vlm(pdf_path: str, pages: list[int], page_images: list[str]) -> str:
    """Extract text using GPT-4o-mini vision."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    texts = []
    for idx, page_b64 in zip(pages, page_images):
        response = llm.invoke([{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Извлеки весь текст с этой страницы PDF точно, включая таблицы и числа. Только текст, без комментариев.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{page_b64}", "detail": "high"},
                },
            ],
        }])
        texts.append(response.content)
    return "\n\n".join(texts)


def extract_docling(pdf_path: str, pages: list[int]) -> str:
    """Extract text using Docling (AI-powered parser)."""
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        text = result.document.export_to_markdown()
        return text[:50000]  # limit
    except ImportError:
        return extract_pdfplumber(pdf_path, pages)
    except Exception as e:
        return f"[Docling error: {e}]"


def extract_pymupdf(pdf_path: str, pages: list[int]) -> str:
    """Extract text using PyMuPDF (fitz)."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        texts = []
        for i in pages:
            if i < len(doc):
                texts.append(doc[i].get_text())
        return "\n\n".join(texts)
    except ImportError:
        return extract_pdfplumber(pdf_path, pages)


EXTRACTORS = {
    "pdfplumber": lambda path, pages, images: extract_pdfplumber(path, pages),
    "vlm": lambda path, pages, images: extract_vlm(path, pages, images),
    "docling": lambda path, pages, images: extract_docling(path, pages),
    "pymupdf": lambda path, pages, images: extract_pymupdf(path, pages),
}


# ─── LangGraph nodes ──────────────────────────────────────────────────────────

def prepare_images(state: ExtractionState) -> ExtractionState:
    """Node 1: convert PDF pages to base64 PNG @ 200 DPI."""
    images = []
    try:
        from pdf2image import convert_from_path
        page_nums = [p + 1 for p in state["pages"]]  # 1-indexed
        pil_images = convert_from_path(
            state["pdf_path"],
            dpi=200,
            first_page=min(page_nums),
            last_page=max(page_nums),
        )
        for img in pil_images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            images.append(base64.b64encode(buf.getvalue()).decode())
    except Exception as e:
        print(f"  [prepare_images] Warning: {e}. Using empty images.")

    # Populate method queue
    methods = state.get("methods") or DEFAULT_METHODS
    return {
        **state,
        "page_images": images,
        "method_queue": list(methods),
        "current_method": methods[0] if methods else "pdfplumber",
        "raw_extractions": {},
        "vl_scores": {},
        "methods_tried": [],
        "attempt": 0,
    }


def run_parser(state: ExtractionState) -> ExtractionState:
    """Node 2: run current extraction method."""
    method = state["current_method"]
    print(f"  [run_parser] method={method}")

    extractor = EXTRACTORS.get(method, EXTRACTORS["pdfplumber"])
    try:
        text = extractor(state["pdf_path"], state["pages"], state["page_images"])
    except Exception as e:
        text = f"[Error in {method}: {e}]"

    raw_extractions = {**state["raw_extractions"], method: text}
    methods_tried = state["methods_tried"] + [method]
    return {**state, "raw_extractions": raw_extractions, "methods_tried": methods_tried}


def vl_reflect(state: ExtractionState) -> ExtractionState:
    """Node 3: VL model compares page image vs extracted text, scores 4 criteria."""
    method = state["current_method"]
    extracted_text = state["raw_extractions"].get(method, "")
    images = state["page_images"]

    if not images:
        # No images — skip visual check, assign score based on text quality
        score = 7.0 if len(extracted_text) > 100 else 3.0
        vl_scores = {
            **state["vl_scores"],
            method: {
                "completeness": score, "structure": score,
                "accuracy": score, "charts": score, "avg_total": score,
                "verdict": "accept" if score >= 7 else "retry",
            },
        }
        print(f"  [vl_reflect] No images, text-based score={score}")
        return {**state, "vl_scores": vl_scores}

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Use first page image for evaluation
    page_b64 = images[0]
    score_prompt = """Сравни оригинальную страницу PDF с извлечённым текстом.
Оцени от 1 до 10 по 4 критериям:
1. completeness — всё ли извлечено?
2. structure — сохранена ли структура (таблицы, абзацы)?
3. accuracy — точны ли числа и слова?
4. charts — корректно ли обработаны графики и таблицы?

Извлечённый текст (первые 800 символов):
{text}

Верни JSON:
{{"completeness": 0-10, "structure": 0-10, "accuracy": 0-10, "charts": 0-10, "verdict": "accept"/"retry", "reason": "..."}}"""

    response = llm.invoke([{
        "role": "user",
        "content": [
            {"type": "text", "text": score_prompt.format(text=extracted_text[:800])},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{page_b64}", "detail": "low"}},
        ],
    }])

    raw = response.content
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            scores = json.loads(match.group())
        except Exception:
            scores = {}
    else:
        scores = {}

    completeness = float(scores.get("completeness", 5))
    structure = float(scores.get("structure", 5))
    accuracy = float(scores.get("accuracy", 5))
    charts = float(scores.get("charts", 5))
    avg_total = (completeness + structure + accuracy + charts) / 4

    all_methods_tried = len(state["methods_tried"]) >= len(state.get("methods") or DEFAULT_METHODS)
    force_accept = state["attempt"] >= 3 or all_methods_tried

    verdict = scores.get("verdict", "retry")
    if avg_total >= 7 or force_accept:
        verdict = "accept"

    updated_scores = {
        **state["vl_scores"],
        method: {
            "completeness": completeness, "structure": structure,
            "accuracy": accuracy, "charts": charts,
            "avg_total": avg_total, "verdict": verdict,
            "reason": scores.get("reason", ""),
        },
    }

    print(f"  [vl_reflect] method={method} avg={avg_total:.1f} verdict={verdict}")
    return {
        **state,
        "vl_scores": updated_scores,
        "attempt": state["attempt"] + 1,
    }


def pick_next_method(state: ExtractionState) -> ExtractionState:
    """Node 4: on retry, rotate to next method."""
    remaining = [m for m in (state.get("methods") or DEFAULT_METHODS)
                 if m not in state["methods_tried"]]
    if not remaining:
        # fallback: reuse best tried method
        next_method = state["methods_tried"][-1] if state["methods_tried"] else "pdfplumber"
    else:
        next_method = remaining[0]

    print(f"  [pick_next_method] switching to method={next_method}")
    return {**state, "current_method": next_method}


def select_best(state: ExtractionState) -> ExtractionState:
    """Node 5: pick method with highest avg_total score."""
    if not state["vl_scores"]:
        best_method = state["methods_tried"][0] if state["methods_tried"] else "pdfplumber"
    else:
        best_method = max(state["vl_scores"], key=lambda m: state["vl_scores"][m].get("avg_total", 0))

    final_text = state["raw_extractions"].get(best_method, "")
    print(f"  [select_best] winner={best_method}, text_len={len(final_text)}")
    return {**state, "best_method": best_method, "final_text": final_text}


def should_retry(state: ExtractionState) -> str:
    """Conditional edge: accept or retry."""
    method = state["current_method"]
    scores = state["vl_scores"].get(method, {})
    verdict = scores.get("verdict", "retry")

    if verdict == "accept":
        return "select_best"

    remaining = [m for m in (state.get("methods") or DEFAULT_METHODS)
                 if m not in state["methods_tried"]]
    if not remaining or state["attempt"] >= 3:
        return "select_best"

    return "pick_next_method"


# ─── Build graph ──────────────────────────────────────────────────────────────

def build_extraction_graph():
    from langgraph.graph import END, StateGraph

    graph = StateGraph(ExtractionState)
    graph.add_node("prepare_images", prepare_images)
    graph.add_node("run_parser", run_parser)
    graph.add_node("vl_reflect", vl_reflect)
    graph.add_node("pick_next_method", pick_next_method)
    graph.add_node("select_best", select_best)

    graph.set_entry_point("prepare_images")
    graph.add_edge("prepare_images", "run_parser")
    graph.add_edge("run_parser", "vl_reflect")
    graph.add_conditional_edges("vl_reflect", should_retry, {
        "select_best": "select_best",
        "pick_next_method": "pick_next_method",
    })
    graph.add_edge("pick_next_method", "run_parser")
    graph.add_edge("select_best", END)

    return graph.compile()


def run_extraction_agent(
    pdf_path: str,
    pages: list[int] | None = None,
    methods: list[str] | None = None,
) -> ExtractionState:
    if pages is None:
        pages = list(range(15))
    if methods is None:
        methods = DEFAULT_METHODS

    app = build_extraction_graph()
    initial: ExtractionState = {
        "pdf_path": pdf_path,
        "pages": pages,
        "methods": methods,
        "method_queue": [],
        "current_method": "",
        "raw_extractions": {},
        "vl_scores": {},
        "page_images": [],
        "attempt": 0,
        "final_text": "",
        "best_method": "",
        "methods_tried": [],
    }
    return app.invoke(initial)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")

    pdf_path = str(Path(__file__).parent.parent.parent / "data" / "bsulp_2024_rus_pages_1_15.pdf")
    print("Running Extraction Agent on pages 0-2...")
    result = run_extraction_agent(pdf_path, pages=[0, 1, 2], methods=["pdfplumber", "vlm"])

    print(f"\nBest method: {result['best_method']}")
    print(f"Methods tried: {result['methods_tried']}")
    print(f"VL Scores:")
    for method, scores in result["vl_scores"].items():
        print(f"  {method}: avg={scores.get('avg_total', 0):.1f} verdict={scores.get('verdict')}")
    print(f"\nFinal text preview:\n{result['final_text'][:500]}")
