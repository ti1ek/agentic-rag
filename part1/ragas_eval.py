"""
Part 1, Stage 2: RAGAS evaluation — ContextPrecision + Faithfulness.
Reads results_vX.json, evaluates with RAGAS, attaches scores to LangFuse traces.
"""

import json
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from langfuse import Langfuse
from ragas import evaluate
from ragas.metrics import context_precision, faithfulness

load_dotenv(Path(__file__).parent.parent / ".env")

ROOT = Path(__file__).parent.parent
langfuse = Langfuse()


def evaluate_results(results_path: Path, score_tag: str = "ragas-v1") -> list[dict]:
    """Evaluate RAG results with RAGAS and attach scores to LangFuse traces."""
    results = json.loads(results_path.read_text())

    dataset = Dataset.from_list([
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["context"],
            "ground_truth": r["ground_truth"],
        }
        for r in results
    ])

    print(f"Running RAGAS evaluation on {len(dataset)} queries...")
    eval_result = evaluate(
        dataset=dataset,
        metrics=[context_precision, faithfulness],
    )

    df = eval_result.to_pandas()

    enriched = []
    for i, (row, result) in enumerate(zip(df.itertuples(), results)):
        cp_raw = getattr(row, "context_precision", None)
        faith_raw = getattr(row, "faithfulness", None)

        cp_score = float(cp_raw) if cp_raw == cp_raw and cp_raw is not None else 0.0
        faith_score = float(faith_raw) if faith_raw == faith_raw and faith_raw is not None else 0.0

        trace_id = result.get("trace_id")
        if trace_id:
            langfuse.create_score(
                trace_id=trace_id,
                name="context_precision",
                value=cp_score,
                comment=score_tag,
            )
            langfuse.create_score(
                trace_id=trace_id,
                name="faithfulness",
                value=faith_score,
                comment=score_tag,
            )

        enriched.append({
            **result,
            "context_precision": cp_score,
            "faithfulness": faith_score,
        })
        print(f"[{i+1:02d}] CP={cp_score:.3f} | Faith={faith_score:.3f} | {result['question'][:50]}...")

    avg_cp = sum(r["context_precision"] for r in enriched) / len(enriched)
    avg_faith = sum(r["faithfulness"] for r in enriched) / len(enriched)
    print(f"\n=== AVERAGES [{score_tag}] ===")
    print(f"Context Precision : {avg_cp:.4f}")
    print(f"Faithfulness      : {avg_faith:.4f}")

    langfuse.flush()

    out_path = results_path.parent / results_path.name.replace("results", "eval")
    out_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2))
    print(f"Enriched results saved to {out_path}")

    return enriched


def print_worst(enriched: list[dict], n: int = 3):
    sorted_by_faith = sorted(enriched, key=lambda r: r["faithfulness"])
    print(f"\n=== {n} WORST BY FAITHFULNESS ===")
    for r in sorted_by_faith[:n]:
        print(f"  Faith={r['faithfulness']:.3f} | CP={r['context_precision']:.3f}")
        print(f"  Q: {r['question']}")
        print(f"  A: {r['answer'][:120]}...")
        print()


def print_comparison_table():
    print("\n=== V1 / V2 / V3 COMPARISON ===")
    print(f"{'Config':<50} | {'Faithfulness':>12} | {'ContextPrecision':>15}")
    print("-" * 83)
    for version, label in [
        ("v1", "V1 baseline (text-embedding-3-small, chunk=500)"),
        ("v2", "V2 improved prompt (text-embedding-3-small, chunk=500)"),
        ("v3", "V3 new embedding (all-MiniLM-L6-v2, chunk=1000)"),
    ]:
        path = ROOT / "part1" / f"eval_{version}.json"
        if path.exists():
            data = json.loads(path.read_text())
            avg_f = sum(r.get("faithfulness", 0) for r in data) / len(data)
            avg_cp = sum(r.get("context_precision", 0) for r in data) / len(data)
            print(f"{label:<50} | {avg_f:>12.4f} | {avg_cp:>15.4f}")
        else:
            print(f"{label:<50} | {'N/A':>12} | {'N/A':>15}")


if __name__ == "__main__":
    for version in ["v1", "v2", "v3"]:
        path = ROOT / "part1" / f"results_{version}.json"
        if path.exists():
            print(f"\n{'='*60}")
            print(f"Evaluating {version}...")
            enriched = evaluate_results(path, score_tag=f"ragas-{version}")
            if version == "v1":
                print_worst(enriched)

    print_comparison_table()
