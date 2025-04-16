"""
Part 1, Stage 3 & 4: Prompt V2 (dynamic fetch) + Embedding experiment (V3).

BEFORE RUNNING STAGE 3:
  1. Open cloud.langfuse.com → Prompt Management
  2. Create prompt named "rag-system-prompt"
  3. Create version 1 (baseline), then version 2 (improved few-shot + strict grounding)
  4. Set V2 as production

Then run: python prompt_management.py 3  (stage 3 — V2 prompt)
          python prompt_management.py 4  (stage 4 — V3 embedding)
          python prompt_management.py    (both stages)
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

load_dotenv(Path(__file__).parent.parent / ".env")

ROOT = Path(__file__).parent.parent
PDF_PATH = str(ROOT / "data" / "bsulp_2024_rus_pages_1_15.pdf")
QUESTIONS_PATH = ROOT / "data" / "ragas_questions.json"

langfuse = Langfuse()

FALLBACK_PROMPT_V2 = (
    "Ты — точный аналитический ассистент. Отвечай ИСКЛЮЧИТЕЛЬНО на основе "
    "предоставленного контекста. Не придумывай факты.\n\n"
    "Правила:\n"
    "1. Используй только информацию из контекста\n"
    "2. Если данных нет — скажи 'В предоставленном контексте нет информации по этому вопросу'\n"
    "3. Цитируй конкретные цифры и факты из текста\n\n"
    "Пример хорошего ответа:\n"
    "Вопрос: Когда основана компания?\n"
    "Контекст: 'фабрика введена в эксплуатацию в декабре 1974 года'\n"
    "Ответ: Фабрика введена в эксплуатацию в декабре 1974 года.\n\n"
    "ЗАПРЕЩЕНО: предположения, обобщения без опоры на контекст."
)


def get_prompt_from_langfuse(prompt_name: str = "rag-system-prompt") -> str:
    """Dynamically fetch prompt from LangFuse Prompt Management."""
    try:
        prompt_obj = langfuse.get_prompt(prompt_name)
        compiled = prompt_obj.compile()
        print(f"Fetched prompt '{prompt_name}' from LangFuse (version {prompt_obj.version})")
        return compiled
    except Exception as e:
        print(f"Could not fetch prompt from LangFuse ({e}), using fallback V2")
        return FALLBACK_PROMPT_V2


def run_rag_with_prompt(
    question: str,
    vectorstore: Chroma,
    system_prompt: str,
    version_tag: str,
) -> dict:
    """Run RAG with custom system prompt, LangFuse v4 tracing."""
    trace_id = langfuse.create_trace_id()

    with langfuse.start_as_current_observation(
        name="rag_query",
        as_type="agent",
        input={"question": question},
        metadata={"version": version_tag},
        trace_context={"trace_id": trace_id},
    ):
        with langfuse.start_as_current_observation(name="retrieval", as_type="retriever", input={"query": question}):
            docs = vectorstore.similarity_search(question, k=5)
            context_chunks = [d.page_content for d in docs]
            langfuse.update_current_span(output={"chunks_count": len(docs)})

        context = "\n\n".join(context_chunks)

        handler = CallbackHandler(trace_context={"trace_id": trace_id})
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, callbacks=[handler])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"},
        ]

        with langfuse.start_as_current_observation(name="generation", as_type="generation"):
            response = llm.invoke(messages)
            answer = response.content
            langfuse.update_current_span(output={"answer": answer[:200]})

    langfuse.flush()

    return {
        "question": question,
        "answer": answer,
        "context": context_chunks,
        "trace_id": trace_id,
        "version": version_tag,
    }


def load_vectorstore_v1() -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=str(ROOT / "part1" / "chroma_v1"),
        embedding_function=embeddings,
        collection_name="bayan_sulu_v1",
    )


def build_vectorstore_minilm(chroma_dir: str) -> Chroma:
    """Build vectorstore with all-MiniLM-L6-v2 + larger chunks."""
    from langchain_community.embeddings import HuggingFaceEmbeddings

    loader = PDFPlumberLoader(PDF_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"V3: {len(chunks)} chunks (chunk_size=1000, overlap=100)")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir,
        collection_name="bayan_sulu_v3",
    )
    return vectorstore


def stage3_prompt_v2():
    """Run all 20 questions with Prompt V2 (dynamic fetch from LangFuse)."""
    print("\n=== STAGE 3: Prompt V2 (dynamic fetch from LangFuse) ===")
    system_prompt = get_prompt_from_langfuse()
    vectorstore = load_vectorstore_v1()
    questions_data = json.loads(QUESTIONS_PATH.read_text())

    results = []
    for i, item in enumerate(questions_data):
        q = item["question"]
        print(f"[{i+1}/20] {q[:60]}...")
        result = run_rag_with_prompt(q, vectorstore, system_prompt, version_tag="v2")
        result["ground_truth"] = item["ground_truth"]
        results.append(result)

    out = ROOT / "part1" / "results_v2.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"V2 results saved to {out}")


def stage4_embedding_v3():
    """Run all 20 questions with MiniLM embeddings + larger chunks."""
    print("\n=== STAGE 4: Embedding V3 (all-MiniLM-L6-v2 + chunk=1000) ===")
    from langchain_community.embeddings import HuggingFaceEmbeddings

    chroma_dir = str(ROOT / "part1" / "chroma_v3")
    if not Path(chroma_dir).exists():
        vectorstore = build_vectorstore_minilm(chroma_dir)
    else:
        print("Loading existing V3 vectorstore...")
        vectorstore = Chroma(
            persist_directory=chroma_dir,
            embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            collection_name="bayan_sulu_v3",
        )

    system_prompt = get_prompt_from_langfuse()
    questions_data = json.loads(QUESTIONS_PATH.read_text())

    results = []
    for i, item in enumerate(questions_data):
        q = item["question"]
        print(f"[{i+1}/20] {q[:60]}...")
        result = run_rag_with_prompt(q, vectorstore, system_prompt, version_tag="v3")
        result["ground_truth"] = item["ground_truth"]
        results.append(result)

    out = ROOT / "part1" / "results_v3.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"V3 results saved to {out}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"

    if stage in ("3", "v2", "all"):
        stage3_prompt_v2()

    if stage in ("4", "v3", "all"):
        stage4_embedding_v3()
