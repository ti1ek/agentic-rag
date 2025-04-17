"""
Part 2, Task 1: Extraction Agent — A2A HTTP service on port 5010.
AgentCard + /a2a endpoint.
"""

import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

# Allow importing agent.py from same directory regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))

load_dotenv(Path(__file__).parent.parent.parent / ".env")

from agent import run_extraction_agent  # noqa: E402

app = FastAPI(title="Extraction Agent A2A Service")


# ─── A2A Protocol models ──────────────────────────────────────────────────────

class AgentSkill(BaseModel):
    name: str
    description: str
    input_schema: dict
    output_schema: dict


class AgentCard(BaseModel):
    name: str
    description: str
    version: str
    url: str
    skills: list[AgentSkill]


class A2ARequest(BaseModel):
    skill: str
    input: dict


class A2AResponse(BaseModel):
    skill: str
    output: dict
    error: str | None = None


# ─── Endpoints ────────────────────────────────────────────────────────────────

AGENT_CARD = AgentCard(
    name="ExtractionAgent",
    description="Extracts text from PDF using multiple methods (pdfplumber, VLM, Docling). Internally uses LangGraph with VL Reflector to select the best extraction result.",
    version="1.0.0",
    url="http://localhost:5010",
    skills=[
        AgentSkill(
            name="extract_pdf",
            description="Extract text from PDF pages using multiple parsers + visual validation",
            input_schema={
                "type": "object",
                "properties": {
                    "pdf_path": {"type": "string", "description": "Absolute path to PDF file"},
                    "pages": {"type": "array", "items": {"type": "integer"}, "description": "0-indexed page numbers"},
                    "methods": {"type": "array", "items": {"type": "string"}, "description": "Extraction methods to try"},
                },
                "required": ["pdf_path"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "final_text": {"type": "string"},
                    "best_method": {"type": "string"},
                    "vl_scores": {"type": "object"},
                    "methods_tried": {"type": "array"},
                },
            },
        )
    ],
)


@app.get("/a2a/agent-card")
async def get_agent_card() -> AgentCard:
    return AGENT_CARD


@app.post("/a2a")
async def handle_a2a(request: A2ARequest) -> A2AResponse:
    if request.skill != "extract_pdf":
        return A2AResponse(skill=request.skill, output={}, error=f"Unknown skill: {request.skill}")

    try:
        pdf_path = request.input.get("pdf_path")
        pages = request.input.get("pages", list(range(15)))
        methods = request.input.get("methods", ["pdfplumber", "vlm", "docling"])

        if not pdf_path:
            return A2AResponse(skill=request.skill, output={}, error="pdf_path is required")

        print(f"[A2A] extract_pdf: path={pdf_path}, pages={pages[:3]}..., methods={methods}")
        result = run_extraction_agent(pdf_path=pdf_path, pages=pages, methods=methods)

        return A2AResponse(
            skill="extract_pdf",
            output={
                "final_text": result["final_text"],
                "best_method": result["best_method"],
                "vl_scores": result["vl_scores"],
                "methods_tried": result["methods_tried"],
            },
        )
    except Exception as e:
        return A2AResponse(skill=request.skill, output={}, error=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ExtractionAgent"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5010)
