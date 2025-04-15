"""
Part 1, Stage 1: Naive RAG pipeline with LangFuse v4 tracing.
Runs all 20 questions and saves results to results_v1.json.
"""

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from langchain_chroma import Chroma
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

ROOT = Path(__file__).parent.parent
PDF_PATH = str(ROOT / "data" / "bsulp_2024_rus_pages_1_15.pdf")
QUESTIONS_PATH = ROOT / "data" / "ragas_questions.json"
RESULTS_PATH = ROOT / "part1" / "results_v1.json"
CHROMA_DIR = str(ROOT / "part1" / "chroma_v1")

langfuse = Langfuse()


def build_vectorstore() -> Chroma:
    """Load PDF, chunk, embed, store in ChromaDB."""
    loader = PDFPlumberLoader(PDF_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks from {len(docs)} pages")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="bayan_sulu_v1",
    )
    print(f"Vectorstore built: {CHROMA_DIR}")
    return vectorstore


def load_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="bayan_sulu_v1",
    )


def run_rag(question: str, vectorstore: Chroma, version_tag: str = "v1") -> dict:
    """Run single RAG query with LangFuse v4 tracing."""
    trace_id = langfuse.create_trace_id()

    with langfuse.start_as_current_observation(
        name="rag_query",
        as_type="agent",
        input={"question": question},
        metadata={"version": version_tag},
        trace_context={"trace_id": trace_id},
    ):
        # Retrieval span
        with langfuse.start_as_current_observation(
            name="retrieval",
            as_type="retriever",
            input={"query": question},
        ):
            docs = vectorstore.similarity_search(question, k=5)
            context_chunks = [d.page_content for d in docs]
            langfuse.update_current_span(
                output={"chunks_count": len(docs), "context_preview": context_chunks[0][:100] if context_chunks else ""},
            )

        context = "\n\n".join(context_chunks)

        # Generation span — use LangFuse CallbackHandler for LLM tracing
        handler = CallbackHandler(trace_context={"trace_id": trace_id})
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, callbacks=[handler])

        system_prompt = (
            "Ты — помощник, отвечающий ТОЛЬКО на основе предоставленного контекста. "
            "Если информации в контексте недостаточно, скажи об этом. "
            "Отвечай кратко и точно."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"},
        ]

        with langfuse.start_as_current_observation(
            name="generation",
            as_type="generation",
            input={"question": question},
        ):
            response = llm.invoke(messages)
            answer = response.content
            langfuse.update_current_span(output={"answer": answer[:200]})

        langfuse.update_current_span(output={"final_answer": answer})

    langfuse.flush()

    return {
        "question": question,
        "answer": answer,
        "context": context_chunks,
        "trace_id": trace_id,
        "version": version_tag,
    }


def main():
    # Build or load vectorstore
    if not Path(CHROMA_DIR).exists():
        vectorstore = build_vectorstore()
    else:
        print("Loading existing vectorstore...")
        vectorstore = load_vectorstore()

    questions_data = json.loads(QUESTIONS_PATH.read_text())
    results = []

    for i, item in enumerate(questions_data):
        question = item["question"]
        ground_truth = item["ground_truth"]
        print(f"[{i+1}/20] {question[:60]}...")

        result = run_rag(question, vectorstore, version_tag="v1")
        result["ground_truth"] = ground_truth
        results.append(result)

    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nDone! Results saved to {RESULTS_PATH}")
    print(f"Total queries: {len(results)}")


if __name__ == "__main__":
    main()
