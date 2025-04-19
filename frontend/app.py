"""
Streamlit demo UI for Agentic RAG system.
Shows full orchestration: agent decisions, LangGraph, alpha tuning, retry logic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "part2"))

import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

ROOT = Path(__file__).parent.parent
PDF_PATH = str(ROOT / "data" / "bsulp_2024_rus_pages_1_15.pdf")
GRAPH_PNG = ROOT / "part2" / "orchestrator_graph.png"

st.set_page_config(
    page_title="Agentic RAG — Баян Сулу",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Agentic RAG — Multi-Agent Orchestration")
st.caption("АО «Баян Сулу» · Годовой отчёт 2024 · nFactorial Module 7")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_demo, tab_graph, tab_ragas = st.tabs(["💬 Demo Q&A", "🗺️ LangGraph", "📊 RAGAS Results"])


# ─── Tab 2: LangGraph visualization ──────────────────────────────────────────
with tab_graph:
    st.subheader("Orchestrator LangGraph")
    if GRAPH_PNG.exists():
        st.image(str(GRAPH_PNG), caption="Main Orchestrator — conditional edges from Reflector Agent")
    else:
        st.code("""
graph TD
  __start__ --> extract
  extract --> chunk
  chunk --> retrieve
  retrieve --> generate
  generate --> reflect
  reflect -. end .-> __end__
  reflect -.-> extract
  reflect -.-> retrieve
  reflect -.-> generate
        """, language="text")

    st.markdown("""
**Условные рёбра из Reflector Agent:**
| Условие | Действие |
|---------|----------|
| Все скоры ≥ 7 | ✅ accept → END |
| Relevance < 7 | 🔄 retry → Retrieval Agent |
| Faithfulness < 7 | 🔄 retry → Generator Agent |
| Мусор в тексте | 🔄 retry → Extraction Agent (A2A) |
| attempt ≥ 3 | ⚠️ force accept → END |
""")


# ─── Tab 3: RAGAS Results ─────────────────────────────────────────────────────
with tab_ragas:
    import json

    st.subheader("Part 1 — RAGAS Evaluation Results")
    st.markdown("""
**Конфигурации:**
- **V1** — baseline: text-embedding-3-small + chunk=500 + базовый промпт
- **V2** — улучшенный промпт (few-shot + strict grounding), загружен из LangFuse API
- **V3** — новые эмбеддинги: all-MiniLM-L6-v2 + chunk=1000
""")

    rows = []
    for version, label in [
        ("v1", "V1 baseline"),
        ("v2", "V2 improved prompt"),
        ("v3", "V3 MiniLM embeddings"),
    ]:
        path = ROOT / "part1" / f"eval_{version}.json"
        if path.exists():
            data = json.loads(path.read_text())
            avg_f = sum(r.get("faithfulness", 0) for r in data) / len(data)
            avg_cp = sum(r.get("context_precision", 0) for r in data) / len(data)
            rows.append({"Config": label, "Faithfulness": round(avg_f, 4), "Context Precision": round(avg_cp, 4)})

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("""
**Вывод:** MiniLM (V3) показал значительно хуже — он плохо работает с русскоязычным текстом.
text-embedding-3-small (OpenAI) значительно превосходит для русского языка.
Улучшенный промпт (V2) не дал значимого роста из-за природы вопросов — они числовые, а не аналитические.
""")

    st.divider()
    st.markdown("🔗 Полные трейсы и скоры — [LangFuse Dashboard](https://cloud.langfuse.com)")


# ─── Tab 1: Demo Q&A ──────────────────────────────────────────────────────────
with tab_demo:
    col_left, col_right = st.columns([2, 1])

    with col_right:
        st.markdown("**Примеры вопросов:**")
        examples = [
            "В каком году основана фабрика?",
            "Кто дистрибьютор Баян Сулу?",
            "Какие новые продукты запустили в 2024?",
            "Почему казахстанские кондитеры проигрывают по цене?",
            "Сколько цехов у предприятия?",
        ]
        for ex in examples:
            if st.button(ex, key=ex, use_container_width=True):
                st.session_state["q"] = ex
                st.rerun()

    with col_left:
        question = st.text_input(
            "Задайте вопрос:",
            value=st.session_state.get("q", ""),
            placeholder="Например: Какие новые продукты запустили в 2024?",
            key="q",
        )
        ask_btn = st.button("🚀 Запустить агентов", type="primary", use_container_width=True)

    if ask_btn and question:
        _run_log = []

        @st.cache_resource(show_spinner=False)
        def get_chunks_and_embeddings():
            import numpy as np
            import pdfplumber
            from langchain_openai import OpenAIEmbeddings
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            with pdfplumber.open(PDF_PATH) as pdf:
                texts = [p.extract_text() or "" for p in pdf.pages]
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = []
            for t in texts:
                chunks.extend(splitter.split_text(t))
            chunks = [c for c in chunks if len(c.strip()) > 30]

            emb = OpenAIEmbeddings(model="text-embedding-3-small")
            vectors = np.array(emb.embed_documents(chunks), dtype="float32")
            return chunks, vectors

        try:
            import numpy as np
            from rank_bm25 import BM25Okapi
            from langchain_openai import OpenAIEmbeddings
            from generator_agent import generate_node, GeneratorState
            from reflector_agent import reflect_node, ReflectorState

            # ── Live agent steps ──────────────────────────────────────────────
            with st.status("🔄 Multi-agent pipeline запущен...", expanded=True) as status:

                st.write("📄 **Extraction Agent** — pdfplumber извлекает текст из PDF...")
                chunks, chunk_embeddings = get_chunks_and_embeddings()
                st.write(f"   ✓ Извлечено и проиндексировано **{len(chunks)} чанков** (метод: pdfplumber, cached)")

                st.write("✂️ **Chunking Agent** — выбрана стратегия `recursive` (chunk=500, overlap=50)")
                st.write(f"   ✓ Итого чанков: **{len(chunks)}**, средняя длина: ~{sum(len(c) for c in chunks)//len(chunks)} символов")

                st.write("🔍 **Retrieval Agent** — тестирую BM25 + Dense + Hybrid с alpha-tuning...")

                # BM25
                tokenized = [c.lower().split() for c in chunks]
                bm25 = BM25Okapi(tokenized)
                bm25_scores = np.array(bm25.get_scores(question.lower().split()), dtype=float)

                # Dense
                emb = OpenAIEmbeddings(model="text-embedding-3-small")
                q_vec = np.array(emb.embed_query(question), dtype=float)
                norms = np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(q_vec) + 1e-9
                dense_scores = chunk_embeddings @ q_vec / norms

                bm25_norm = bm25_scores / (bm25_scores.max() + 1e-9)
                dense_norm = dense_scores / (dense_scores.max() + 1e-9)

                # Alpha tuning results
                alpha_results = {}
                for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
                    hybrid = alpha * dense_norm + (1 - alpha) * bm25_norm
                    top5 = np.argsort(hybrid)[::-1][:5]
                    avg_score = float(hybrid[top5].mean())
                    alpha_results[alpha] = {"score": avg_score, "top_idx": top5}

                best_alpha = max(alpha_results, key=lambda a: alpha_results[a]["score"])
                final_idx = alpha_results[best_alpha]["top_idx"]
                final_docs = [chunks[i] for i in final_idx]

                search_type = "pure BM25" if best_alpha == 0.0 else ("pure dense (vector)" if best_alpha == 1.0 else f"hybrid (α={best_alpha})")
                st.write(f"   ✓ Лучший α = **{best_alpha}** → **{search_type}** выбран")

                st.write("💬 **Generator Agent** — формирую ответ строго по контексту...")
                gen_state: GeneratorState = {
                    "question": question,
                    "context_chunks": final_docs,
                    "answer": "", "confidence": 0.0, "sources_used": [],
                }
                gen_result = generate_node(gen_state)
                st.write(f"   ✓ Ответ сформирован, уверенность: **{gen_result['confidence']:.0%}**")

                st.write("🔎 **Reflector Agent** — оцениваю по 3 критериям...")
                ref_state: ReflectorState = {
                    "question": question,
                    "context_chunks": final_docs,
                    "answer": gen_result["answer"],
                    "confidence": gen_result["confidence"],
                    "attempt": 0,
                    "scores": {}, "decision": "", "feedback": "", "retry_target": "",
                }
                ref_result = reflect_node(ref_state)
                scores = ref_result["scores"]
                decision = ref_result["decision"]
                st.write(f"   ✓ Relevance={scores.get('relevance',0):.0f} | Faithfulness={scores.get('faithfulness',0):.0f} | Completeness={scores.get('completeness',0):.0f} → **{decision}**")

                status.update(label="✅ Все агенты завершили работу!", state="complete", expanded=False)

            # ── Result ────────────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 💬 Ответ")
            st.success(gen_result["answer"])

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Уверенность", f"{gen_result['confidence']:.0%}")
            m2.metric("Relevance", f"{scores.get('relevance', 0):.0f}/10")
            m3.metric("Faithfulness", f"{scores.get('faithfulness', 0):.0f}/10")
            m4.metric("Completeness", f"{scores.get('completeness', 0):.0f}/10")

            if decision == "accept":
                st.info("✅ Reflector принял ответ с первой попытки")
            else:
                st.warning(f"🔄 Reflector решил: **{decision}** — retry")

            # Alpha tuning log
            with st.expander("📈 Alpha Tuning Log (Retrieval Agent)"):
                alpha_rows = [
                    {"Alpha": a, "Score": round(v["score"], 4),
                     "Тип": "pure BM25" if a == 0.0 else ("pure dense" if a == 1.0 else "hybrid"),
                     "Выбран": "✅" if a == best_alpha else ""}
                    for a, v in alpha_results.items()
                ]
                import pandas as pd
                st.dataframe(pd.DataFrame(alpha_rows), hide_index=True, use_container_width=True)

            with st.expander("📄 Найденные фрагменты (top-5)"):
                for i, doc in enumerate(final_docs):
                    st.markdown(f"**Чанк {i+1}:**")
                    st.text(doc[:300] + ("..." if len(doc) > 300 else ""))

        except Exception as e:
            st.error(f"Ошибка: {e}")
            st.exception(e)

    elif ask_btn:
        st.warning("Введите вопрос.")

st.divider()
st.caption("nFactorial AI Engineering · Module 7 · Agentic RAG · LangFuse + RAGAS + LangGraph")
