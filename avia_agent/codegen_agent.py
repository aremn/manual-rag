import json
import os
import time
import traceback
from typing import Any

from openai import OpenAI

from .codegen_runtime import lookup_xlsx_qa_row, read_pdf_pages, ugrep_search


def _load_api_key() -> str | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key and api_key.startswith("{") and api_key.endswith("}"):
        return api_key[1:-1]
    return api_key


def _extract_completion_text(message) -> str:
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            text = getattr(part, "text", None)
            parts.append(text if text else str(part))
        return "\n".join(parts)
    return str(message)


def _extract_python_code(text: str) -> str:
    if "```python" in text:
        return text.split("```python", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    return text.strip()


def _write_jsonl(path: str, payload: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


SYSTEM_PROMPT = """
You are a code-writing QA agent.

Goal: answer the user's question using ONLY the local manuals folder (PDF files). You must generate Python code that searches the manuals via the CLI tool `ugrep` using `--filter` (file-first approach).

IMPORTANT: The helper functions below are ALREADY AVAILABLE in the global scope. DO NOT import them!
Just call them directly like: `ugrep_search(...)`, `read_pdf_pages(...)`, etc.

Available helper functions (no import needed):
- `ugrep_search(docs_root, query, mode='fixed', extensions=('pdf',), max_results=30) -> list`
  - This runs `ugrep --filter ...` under the hood and returns parsed matches: [{path, location, snippet}, ...]
  - The search is case-insensitive.
- `read_pdf_pages(path, page_start, page_end) -> str`

Rules:
1) ALWAYS search PDFs: use extensions=('pdf',). This is the primary source.

2) CRITICAL for search queries:
   - Use SHORT, SPECIFIC search terms (2-3 words max)
   - Search for the MAIN SUBJECT of the question, not words like "purpose", "explain", "what is"
   - Example: "Explain the purpose of the Training Manual" -> search for "Training Manual", "OM-D", "Part D Training"
   - ALWAYS search for SYNONYMS and ALTERNATIVE PHRASINGS:
     * "Training Manual" -> also search "OM-D", "Part D", "training programme"
     * "inexperienced" -> also search "low experience", "crewing inexperienced"
   - Do 3-5 targeted searches with different terms

3) CRITICAL for answer synthesis:
   - DO NOT filter pages based on question words like "purpose", "explain", "define"!
   - The document DESCRIBES things without using those meta-words
   - Example: Q asks for "purpose of Training Manual" but doc says "Training Manual contains..." - this IS the answer!
   - If a page DESCRIBES or DEFINES the subject, it answers questions about purpose/scope/objective
   - Read pages that mention the SUBJECT (Training Manual, OM-D, etc.) and extract their description

4) CRITICAL for citations - FILTER for relevance:
   - After collecting matches, keep only pages that DESCRIBE the subject
   - Return MAX 5-10 citations that DIRECTLY answer the question
   - A page mentioning a keyword once in passing is NOT relevant

5) `ugrep_search` returns `location` as `page N` for PDFs.
6) Never invent facts. If not found: return "Not found in the provided manuals".

Output format - return ONLY a Python code block:

    def solve(docs_root: str, question: str) -> dict:
        # Search for key terms
        results = ugrep_search(docs_root, "key term", extensions=("pdf",))
        # Read relevant pages and extract answer
        ...
        return {
            "answer": "concise answer based on manual content",
            "citations": [{"path": "...", "location": "page N", "quote": "..."}]
        }
"""


def answer_with_codegen(
    *,
    model: str,
    docs_root: str,
    question: str,
    max_retries: int = 3,
    log_path: str | None = None,
    verbose: bool = False,
    show_code: bool = False,
    enable_fallback: bool = True,
) -> dict[str, Any]:
    api_key = _load_api_key()
    if not api_key:
        if enable_fallback:
            fb = _fallback_xlsx(docs_root, question, verbose=verbose, log_path=log_path)
            if fb:
                return fb
        return {"answer": "OPENAI_API_KEY is not set.", "citations": []}
    client = OpenAI(api_key=api_key)

    docs_root = os.path.abspath(docs_root)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": f"Docs root: {docs_root}\nQuestion: {question}"},
    ]

    if log_path:
        _write_jsonl(
            log_path,
            {
                "event": "start",
                "model": model,
                "docs_root": docs_root,
                "question": question,
            },
        )
    if verbose:
        print(f"[codegen] docs_root={docs_root}")
        print(f"[codegen] model={model}")
        print(f"[codegen] question={question}")
        if log_path:
            print(f"[codegen] log_path={log_path}")

    class SearchResult:
        def __init__(self, payload: dict[str, Any]):
            self._payload = payload

        def get(self, key: str, default: Any = None) -> Any:
            return self._payload.get(key, default)

        @property
        def payload(self) -> dict[str, Any]:
            return self._payload

        @property
        def results(self) -> list[dict]:
            return list(self._payload.get("results") or [])

        def __iter__(self):
            return iter(self.results)

        def __len__(self) -> int:
            return len(self.results)

        def __repr__(self) -> str:
            return f"SearchResult(results={len(self)})"

    def traced_ugrep_search(*args, **kwargs):
        docs_root_arg = kwargs.pop("docs_root", None)
        query_arg = kwargs.pop("query", None)
        if docs_root_arg is None and len(args) >= 1:
            docs_root_arg = args[0]
        if query_arg is None and len(args) >= 2:
            query_arg = args[1]
        mode = kwargs.pop("mode", "fixed")
        extensions = kwargs.pop("extensions", ("pdf",))
        max_results = int(kwargs.pop("max_results", 30))

        payload = ugrep_search(
            docs_root=str(docs_root_arg),
            query=str(query_arg),
            mode=str(mode),
            extensions=extensions,
            max_results=max_results,
        )
        if log_path:
            _write_jsonl(log_path, {"event": "ugrep_search", "payload": payload})
        if verbose:
            n = len(payload.get("results") or [])
            print(
                f"[ugrep] query={str(query_arg)!r} exts={list(extensions)} ok={payload.get('ok')} results={n}"
            )
            for p in (payload.get("passes") or [])[:10]:
                print(
                    f"[ugrep] pass exts={p.get('only_exts')} rc={p.get('returncode')} dt={p.get('duration_sec'):.2f}s lines={p.get('stdout_lines')}"
                )
        return SearchResult(payload)

    def traced_read_pdf_pages(path: str, page_start: int, page_end: int) -> str:
        if verbose:
            print(f"[pdf] read {path} pages {page_start}-{page_end}")
        txt = read_pdf_pages(path, page_start, page_end)
        if log_path:
            _write_jsonl(
                log_path,
                {
                    "event": "read_pdf_pages",
                    "path": path,
                    "page_start": page_start,
                    "page_end": page_end,
                },
            )
        return txt

    def traced_lookup_xlsx_qa_row(path: str, question: str):
        row = lookup_xlsx_qa_row(path, question)
        if log_path:
            _write_jsonl(
                log_path,
                {"event": "lookup_xlsx_qa_row", "path": path, "found": bool(row)},
            )
        if verbose:
            print(f"[xlsx] lookup row in {path}: found={bool(row)}")
        return row

    for attempt in range(1, max_retries + 1):
        started = time.time()
        if verbose:
            print(f"[codegen] attempt {attempt}/{max_retries}: generating solve()...")
        completion = client.chat.completions.create(model=model, messages=messages, temperature=0.2)
        text = _extract_completion_text(completion.choices[0].message)
        code = _extract_python_code(text)

        if log_path:
            _write_jsonl(
                log_path,
                {
                    "event": "llm",
                    "attempt": attempt,
                    "duration_sec": time.time() - started,
                    "raw": text,
                    "code": code,
                },
            )
        if show_code or verbose:
            print(f"[codegen] generated code (attempt {attempt}):\n{code}\n")

        env: dict[str, Any] = {
            "ugrep_search": traced_ugrep_search,
            "read_pdf_pages": traced_read_pdf_pages,
            "lookup_xlsx_qa_row": traced_lookup_xlsx_qa_row,
        }

        try:
            exec(code, env)
            if "solve" not in env or not callable(env["solve"]):
                raise RuntimeError("No solve(docs_root, question) function found.")
            if verbose:
                print("[codegen] running solve()...")
            result = env["solve"](docs_root, question)
            if not isinstance(result, dict) or "answer" not in result:
                raise RuntimeError("solve() must return a dict with at least an 'answer' key.")
            result.setdefault("citations", [])
            if log_path:
                _write_jsonl(log_path, {"event": "result", "attempt": attempt, "result": result})
            if verbose:
                print(
                    f"[codegen] solve() returned answer_len={len(result.get('answer') or '')} citations={len(result.get('citations') or [])}"
                )
            return result
        except Exception as e:
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            if log_path:
                _write_jsonl(
                    log_path,
                    {
                        "event": "error",
                        "attempt": attempt,
                        "error": str(e),
                        "traceback": tb,
                    },
                )
            if verbose:
                print(f"[codegen] attempt {attempt} failed: {e}")
            if attempt >= max_retries:
                if enable_fallback:
                    if verbose:
                        print("[codegen] using fallback: ugrep xlsx -> exact row lookup")
                    fb = _fallback_xlsx(docs_root, question, verbose=verbose, log_path=log_path)
                    if fb:
                        return fb
                return {
                    "answer": f"Not found in the provided manuals (codegen failed: {e}).",
                    "citations": [],
                }
            messages.append({"role": "assistant", "content": f"```python\n{code}\n```"})
            messages.append(
                {
                    "role": "user",
                    "content": "The code failed with this exception/traceback. Fix the code and return only corrected Python.\n\n"
                    + tb,
                }
            )

    return {"answer": "Not found in the provided manuals.", "citations": []}


def _fallback_xlsx(
    docs_root: str, question: str, *, verbose: bool, log_path: str | None
) -> dict[str, Any] | None:
    sr = ugrep_search(
        docs_root=docs_root,
        query=question,
        mode="fixed",
        extensions=("xlsx",),
        max_results=20,
    )
    if log_path:
        _write_jsonl(log_path, {"event": "fallback_search", "payload": sr})
    if verbose:
        print(f"[fallback] ugrep ok={sr.get('ok')} results={len(sr.get('results') or [])}")
        for p in (sr.get("passes") or [])[:10]:
            print(
                f"[fallback] pass exts={p.get('only_exts')} rc={p.get('returncode')} dt={p.get('duration_sec'):.2f}s lines={p.get('stdout_lines')}"
            )

    for hit in sr.get("results") or []:
        path = hit.get("path") or ""
        if not path.lower().endswith(".xlsx"):
            continue
        try:
            row = lookup_xlsx_qa_row(path, question)
        except Exception:
            row = None
        if row:
            return {
                "answer": row.to_tsv(),
                "citations": [
                    {
                        "path": path,
                        "location": hit.get("location", ""),
                        "quote": hit.get("snippet", ""),
                    }
                ],
            }
    return None
