import argparse
import json
import random
import re
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def extract_answer_letter(output: str) -> Optional[str]:
    if not output:
        return None
    patterns = [
        r"The answer is\s+([A-Z])\b",
        r"Answer:\s*(?:Option\s*)?([A-Z])\b",
        r"Final Answer.*?\b([A-Z])\b",
    ]
    for p in patterns:
        m = re.search(p, output, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).upper()
    tail = output[-200:]
    m = re.search(r"\b([A-J])\b", tail, re.IGNORECASE)
    return m.group(1).upper() if m else None


def extract_dag_edges(output: str) -> List[Tuple[str, str]]:
    if not output:
        return []
    edges: List[Tuple[str, str]] = []

    # MedVerse <Outline> style
    outline_lines = re.findall(
        r"<Outline>\s*Transient Step\s+\d+:\s*(.*?)\s*;\s*Dependency:",
        output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for line in outline_lines:
        if "->" in line:
            parts = [x.strip() for x in line.split("->")]
            for i in range(len(parts) - 1):
                if parts[i] and parts[i + 1]:
                    edges.append((parts[i], parts[i + 1]))

    # "Finding Reasoning Path:" style with numbered lines
    path_lines = re.findall(
        r"^\s*\d+\s*[:：]\s*(.+)$",
        output,
        flags=re.MULTILINE,
    )
    for line in path_lines:
        line = line.replace("→", "->")
        if "->" in line:
            parts = [x.strip() for x in line.split("->")]
            for i in range(len(parts) - 1):
                if parts[i] and parts[i + 1]:
                    edges.append((parts[i], parts[i + 1]))

    # Markdown bullet style in baseline logs
    md_path_lines = re.findall(
        r"^\s*\d+\.\s*\*\*.*?\*\*\s*→\s*\*\*.*?\*\*.*$",
        output,
        flags=re.MULTILINE,
    )
    for line in md_path_lines:
        clean = re.sub(r"\*\*", "", line).replace("→", "->")
        segs = [x.strip(" .-") for x in clean.split("->")]
        for i in range(len(segs) - 1):
            if segs[i] and segs[i + 1]:
                edges.append((segs[i], segs[i + 1]))

    # de-dup while preserving order
    seen = set()
    uniq: List[Tuple[str, str]] = []
    for h, t in edges:
        key = (normalize_text(h), normalize_text(t))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((h, t))
    return uniq


def option_text(sample: Dict[str, Any], letter: str) -> str:
    options = sample.get("options", {})
    return str(options.get(letter, "")).strip()


@dataclass
class Case:
    case_id: str
    question: str
    answer_idx: str
    answer_text: str
    medverse_output: str
    baseline_output: str
    medverse_pred: Optional[str]
    baseline_pred: Optional[str]
    medverse_correct: bool
    baseline_correct: bool
    medverse_edges: List[Tuple[str, str]]
    baseline_edges: List[Tuple[str, str]]


def build_cases(
    medverse_json_path: str,
    baseline_json_path: str,
    eval_data_jsonl_path: str,
) -> List[Case]:
    medverse = load_json(medverse_json_path)
    baseline = load_json(baseline_json_path)
    eval_data = load_jsonl(eval_data_jsonl_path)

    hybrid_samples = medverse["result"]["hybrid_samples"]

    # Build robust baseline lookup for different schemas:
    # - MedXpert logs use "id" like "Text-1001"
    # - HLE logs use "idx" (int)
    baseline_by_key: Dict[str, Dict[str, Any]] = {}
    baseline_by_question: Dict[str, Dict[str, Any]] = {}
    for row in baseline:
        if not isinstance(row, dict):
            continue
        if "id" in row and row["id"] is not None:
            baseline_by_key[str(row["id"])] = row
        if "idx" in row and row["idx"] is not None:
            baseline_by_key[str(row["idx"])] = row
        q = row.get("question")
        if isinstance(q, str) and q.strip():
            baseline_by_question[normalize_text(q)] = row

    eval_by_idx: Dict[int, Dict[str, Any]] = {}
    eval_by_id: Dict[str, Dict[str, Any]] = {}
    for row in eval_data:
        if not isinstance(row, dict):
            continue
        if "idx" in row:
            try:
                eval_by_idx[int(row["idx"])] = row
            except Exception:
                pass
        if "id" in row and row["id"] is not None:
            eval_by_id[str(row["id"])] = row

    cases: List[Case] = []
    for i, hs in enumerate(hybrid_samples):
        sid = hs.get("id", i)
        if not isinstance(sid, int):
            try:
                sid_int = int(sid)
            except Exception:
                sid_int = None
        else:
            sid_int = sid

        # Resolve eval row:
        # 1) by idx field (works for HLE where ids are sparse like 3,4,5,...)
        # 2) by position (works for MedXpert where hybrid id often equals row index)
        # 3) by id string
        row: Optional[Dict[str, Any]] = None
        if sid_int is not None and sid_int in eval_by_idx:
            row = eval_by_idx[sid_int]
        elif sid_int is not None and 0 <= sid_int < len(eval_data):
            row = eval_data[sid_int]
        elif str(sid) in eval_by_id:
            row = eval_by_id[str(sid)]
        if row is None:
            continue

        cid = row.get("id")
        idx_val = row.get("idx")
        if cid is None:
            cid = f"idx-{idx_val}" if idx_val is not None else f"case-{i}"

        # Resolve baseline row by multiple keys, then question-text fallback.
        base_row = None
        for key in [str(cid), str(idx_val) if idx_val is not None else None, str(sid)]:
            if key and key in baseline_by_key:
                base_row = baseline_by_key[key]
                break
        if base_row is None:
            qkey = normalize_text(row.get("question", ""))
            base_row = baseline_by_question.get(qkey)
        if not base_row:
            continue

        gold = str(row.get("answer_idx", "")).strip().upper()[:1]
        medverse_output = hs.get("output", "") or ""
        baseline_output = base_row.get("output", "") or ""
        med_pred = extract_answer_letter(medverse_output)
        base_pred = extract_answer_letter(baseline_output)

        cases.append(
            Case(
                case_id=cid,
                question=row.get("question", ""),
                answer_idx=gold,
                answer_text=option_text(row, gold),
                medverse_output=medverse_output,
                baseline_output=baseline_output,
                medverse_pred=med_pred,
                baseline_pred=base_pred,
                medverse_correct=(med_pred == gold),
                baseline_correct=(base_pred == gold),
                medverse_edges=extract_dag_edges(medverse_output),
                baseline_edges=extract_dag_edges(baseline_output),
            )
        )
    return cases


def parse_json_obj(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return json.loads(m.group(0))


def strip_tags(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\*\*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def canonicalize_reasoning_for_judge(reasoning_text: str, edges: List[Tuple[str, str]]) -> str:
    """Normalize different output formats into a comparable representation."""
    steps: List[str] = []

    # MedVerse execution steps
    step_blocks = re.findall(r"<Step>\s*(.*?)\s*</Step>", reasoning_text, flags=re.DOTALL | re.IGNORECASE)
    for blk in step_blocks:
        cleaned = strip_tags(blk)
        if cleaned:
            steps.append(cleaned)

    # Markdown style numbered reasoning
    if not steps:
        numbered = re.findall(r"^\s*\d+\.\s*(.+)$", reasoning_text, flags=re.MULTILINE)
        for line in numbered:
            cleaned = strip_tags(line)
            if cleaned:
                steps.append(cleaned)

    # Fallback: sentence chunks
    if not steps:
        chunks = re.split(r"(?<=[\.\!\?])\s+", strip_tags(reasoning_text))
        steps = [c.strip() for c in chunks if c.strip()]

    # Keep a bounded number of steps for stable judging.
    steps = steps[:12]
    step_text = "\n".join([f"- {s}" for s in steps]) if steps else "- (no parsed step)"
    edge_text = "\n".join([f"- {h} -> {t}" for h, t in edges]) if edges else "- (no edges parsed)"
    final_ans = extract_answer_letter(reasoning_text)

    return (
        f"Parsed Edges:\n{edge_text}\n\n"
        f"Normalized Reasoning Steps:\n{step_text}\n\n"
        f"Predicted Option: {final_ans if final_ans else 'UNKNOWN'}"
    )


def eval_reasoning_with_llm(
    client: OpenAI,
    model_name: str,
    question: str,
    gold_answer: str,
    reasoning_text: str,
    edges: List[Tuple[str, str]],
) -> Dict[str, Any]:
    normalized_reasoning = canonicalize_reasoning_for_judge(reasoning_text, edges)
    prompt = f"""
You are a strict medical reasoning evaluator.
Given:
1) A medical multiple-choice question
2) The gold reference answer from physicians
3) A model's reasoning graph/text

Important fairness rule:
- Do NOT reward or penalize output style/format (XML tags, markdown headings, verbosity).
- Judge only medical causal logic, evidence grounding, and clinical risk.
- Use the normalized representation below as primary evidence.

Please score the following dimensions:
- Causal Structure Correctness: node-level causal plausibility and logical jumps
- Verifiability: whether each step is evidence-grounded in question text or mainstream medical knowledge
- Expert-style quality: completeness, conciseness, logical soundness
- Clinical risk if this reasoning leads to a wrong decision

Return ONLY valid JSON with keys:
{{
  "causal_validity_score": int (1-5),
  "causal_edge_correct_count": int (>=0),
  "causal_edge_total": int (>=0),
  "logical_jump_count": int (>=0),
  "evidence_grounding_score": int (1-5),
  "hallucination_count": int (>=0),
  "completeness": int (1-5),
  "conciseness": int (1-5),
  "logical_soundness": int (1-5),
  "risk_level": int (1 or 3 or 5),
  "risk_rationale": string
}}

Risk rubric:
- 1: Inconsequential
- 3: Misleading
- 5: Fatal/Critical

Question:
{question}

Gold answer:
{gold_answer}

Normalized model reasoning package:
{normalized_reasoning}
"""
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Return strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    content = resp.choices[0].message.content or "{}"
    return parse_json_obj(content)


def mos(v: Dict[str, Any]) -> Optional[float]:
    vals = [v.get("completeness"), v.get("conciseness"), v.get("logical_soundness")]
    vals = [x for x in vals if isinstance(x, (int, float))]
    return mean(vals) if vals else None


def fmt_score(x: Any) -> str:
    if x is None:
        return "None"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def summarize_model(records: List[Dict[str, Any]], correctness: List[bool]) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"count": 0}
    cv = [r.get("causal_validity_score") for r in records if isinstance(r.get("causal_validity_score"), (int, float))]
    eg = [r.get("evidence_grounding_score") for r in records if isinstance(r.get("evidence_grounding_score"), (int, float))]
    hc = [r.get("hallucination_count", 0) for r in records if isinstance(r.get("hallucination_count"), (int, float))]
    ljs = [r.get("logical_jump_count", 0) for r in records if isinstance(r.get("logical_jump_count"), (int, float))]
    moses = [mos(r) for r in records if mos(r) is not None]
    edge_correct = sum(int(r.get("causal_edge_correct_count", 0)) for r in records)
    edge_total = sum(int(r.get("causal_edge_total", 0)) for r in records)

    incorrect_indices = [i for i, ok in enumerate(correctness) if not ok]
    high_risk_count = 0
    for i in incorrect_indices:
        if records[i].get("risk_level") == 5:
            high_risk_count += 1
    hall_free = sum(1 for r in records if int(r.get("hallucination_count", 0)) == 0)
    return {
        "count": n,
        "causal_validity_score_avg": mean(cv) if cv else None,
        "edge_accuracy": (edge_correct / edge_total) if edge_total > 0 else None,
        "evidence_grounding_score_avg": mean(eg) if eg else None,
        "hallucination_free_rate": hall_free / n,
        "avg_hallucination_count": mean(hc) if hc else None,
        "avg_logical_jump_count": mean(ljs) if ljs else None,
        "mos": mean(moses) if moses else None,
        "high_risk_error_rate": (
            high_risk_count / len(incorrect_indices) if incorrect_indices else None
        ),
        "incorrect_count": len(incorrect_indices),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--medverse_json",
        type=str,
        default="/home/jimchen/MedVerse/inference/inference_engine/benchmark_results0/HLE_biomed.json",
    )
    parser.add_argument(
        "--baseline_json",
        type=str,
        default="/home/jimchen/MedReason/results/MedReason-8B-results/logs/MedReason-8BHLE_biomedl_api_strict-prompt.json",
    )
    parser.add_argument(
        "--eval_data_jsonl",
        type=str,
        default="/home/jimchen/MedReason/eval_data/HLE_biomed.jsonl",
    )
    parser.add_argument("--judge_model", type=str, default="gpt-5.2")
    parser.add_argument("--api_base", type=str, default="https://openai-api.shenmishajing.workers.dev/v1")
    parser.add_argument("--api_key", type=str, default="aB7cD9eF2gH5iJ8kL1mN4oP6qR3sT0uV")
    parser.add_argument("--max_cases", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--sample_mode", type=str, default="random", choices=["random", "head"])
    parser.add_argument(
        "--output_json",
        type=str,
        default="/home/jimchen/MedVerse/inference/inference_engine/reasoning_quality_compare_HLE_biomed.json",
    )
    args = parser.parse_args()

    client_kwargs: Dict[str, Any] = {}
    if args.api_base:
        client_kwargs["base_url"] = args.api_base
    if args.api_key:
        client_kwargs["api_key"] = args.api_key
    client = OpenAI(**client_kwargs)

    cases = build_cases(args.medverse_json, args.baseline_json, args.eval_data_jsonl)
    if not cases:
        raise RuntimeError("No aligned cases found between MedVerse and baseline logs.")

    if args.sample_mode == "head":
        chosen = cases[: args.max_cases]
    else:
        random.seed(args.seed)
        n = min(args.max_cases, len(cases))
        chosen = random.sample(cases, n)

    medverse_eval: List[Dict[str, Any]] = []
    baseline_eval: List[Dict[str, Any]] = []
    medverse_correctness: List[bool] = []
    baseline_correctness: List[bool] = []

    per_case: List[Dict[str, Any]] = []
    for idx, c in enumerate(chosen, start=1):
        print(f"[{idx}/{len(chosen)}] Evaluating {c.case_id}")
        gold_answer = f"{c.answer_idx}. {c.answer_text}"
        me = eval_reasoning_with_llm(
            client=client,
            model_name=args.judge_model,
            question=c.question,
            gold_answer=gold_answer,
            reasoning_text=c.medverse_output,
            edges=c.medverse_edges,
        )
        be = eval_reasoning_with_llm(
            client=client,
            model_name=args.judge_model,
            question=c.question,
            gold_answer=gold_answer,
            reasoning_text=c.baseline_output,
            edges=c.baseline_edges,
        )

        medverse_eval.append(me)
        baseline_eval.append(be)
        medverse_correctness.append(c.medverse_correct)
        baseline_correctness.append(c.baseline_correct)

        per_case.append(
            {
                "id": c.case_id,
                "gold_answer_idx": c.answer_idx,
                "medverse_correct": c.medverse_correct,
                "baseline_correct": c.baseline_correct,
                "medverse_eval": me,
                "baseline_eval": be,
            }
        )
        print(
            "  MedVerse: "
            f"causal={fmt_score(me.get('causal_validity_score'))}, "
            f"evidence={fmt_score(me.get('evidence_grounding_score'))}, "
            f"mos={fmt_score(mos(me))}, "
            f"risk={fmt_score(me.get('risk_level'))}, "
            f"correct={c.medverse_correct}"
        )
        print(
            "  Baseline: "
            f"causal={fmt_score(be.get('causal_validity_score'))}, "
            f"evidence={fmt_score(be.get('evidence_grounding_score'))}, "
            f"mos={fmt_score(mos(be))}, "
            f"risk={fmt_score(be.get('risk_level'))}, "
            f"correct={c.baseline_correct}"
        )

    both_wrong_indices = [
        i for i in range(len(chosen)) if (not medverse_correctness[i] and not baseline_correctness[i])
    ]
    medverse_both_wrong_high_risk = sum(
        1 for i in both_wrong_indices if medverse_eval[i].get("risk_level") == 5
    )
    baseline_both_wrong_high_risk = sum(
        1 for i in both_wrong_indices if baseline_eval[i].get("risk_level") == 5
    )

    summary = {
        "meta": {
            "judge_model": args.judge_model,
            "sample_size": len(chosen),
            "total_aligned_cases": len(cases),
            "sample_mode": args.sample_mode,
        },
        "medverse_summary": summarize_model(medverse_eval, medverse_correctness),
        "baseline_summary": summarize_model(baseline_eval, baseline_correctness),
        "both_wrong_risk_comparison": {
            "both_wrong_count": len(both_wrong_indices),
            "medverse_high_risk_rate": (
                medverse_both_wrong_high_risk / len(both_wrong_indices)
                if both_wrong_indices
                else None
            ),
            "baseline_high_risk_rate": (
                baseline_both_wrong_high_risk / len(both_wrong_indices)
                if both_wrong_indices
                else None
            ),
        },
    }

    out = {"summary": summary, "per_case": per_case}
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n===== Summary =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {args.output_json}")


if __name__ == "__main__":
    main()
