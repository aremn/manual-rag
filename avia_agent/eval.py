import json
import os
from collections.abc import Iterable
from typing import Any

from .agent import answer_question


def _iter_jsonl(path: str) -> Iterable[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def run_eval(
    *,
    docs_root: str,
    input_path: str,
    out_path: str,
    log_dir: str,
    model: str,
    max_steps: int,
    limit: int = 0,
) -> dict[str, Any]:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    total = 0
    exact = 0
    wrote = 0

    with open(out_path, "a", encoding="utf-8") as out:
        for row in _iter_jsonl(input_path):
            if limit and total >= limit:
                break
            total += 1

            q = row.get("question") or row.get("q") or ""
            gold = row.get("answer") or row.get("a") or ""
            if not q:
                continue

            log_path = os.path.join(log_dir, f"q{total:05d}.jsonl")
            pred = answer_question(
                model=model,
                docs_root=docs_root,
                question=q,
                max_steps=max_steps,
                log_path=log_path,
            )
            pred_text = pred.get("answer", "")
            ok = _norm(pred_text) == _norm(gold) if gold else False
            exact += 1 if ok else 0

            out.write(
                json.dumps(
                    {
                        "idx": total,
                        "question": q,
                        "gold": gold,
                        "pred": pred_text,
                        "citations": pred.get("citations", []),
                        "exact_match": ok,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            wrote += 1

    return {
        "items_total": total,
        "items_written": wrote,
        "exact_match_count": exact,
        "exact_match_rate": (exact / total) if total else 0.0,
        "out_path": out_path,
    }
