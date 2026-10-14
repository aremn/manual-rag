import json
import os
import time
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

from .tools import (
    ListFiles,
    ReadPdfPages,
    ReadTextFile,
    Search,
    ToolResult,
    Tree,
    dispatch_tool,
)


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
        parts: list[str] = []
        for part in content:
            text = getattr(part, "text", None)
            parts.append(text if text else str(part))
        return "\n".join(parts)
    return str(message)


def _write_jsonl(path: str, payload: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


class FinalizeAnswer(BaseModel):
    answer: str = Field(..., description="Final answer to the user. If not found, say so plainly.")
    citations: list[dict[str, str]] = Field(
        default_factory=list,
        description="List of evidence items. Each item should include at least 'path' and optionally 'location' and 'quote'.",
    )


class NextStep(BaseModel):
    current_state: str = Field(..., description="Very short state summary.")
    plan_remaining_steps_brief: list[str] = Field(
        ...,
        description="Remaining plan steps; the first one will be executed now.",
        min_length=1,
        max_length=5,
    )
    task_completed: bool = Field(..., description="True only when ready to answer.")
    function: Tree | ListFiles | Search | ReadTextFile | ReadPdfPages | FinalizeAnswer = Field(
        ..., description="Execute the first remaining step via a tool, or finalize."
    )


SYSTEM_PROMPT = """
You answer questions using ONLY the provided local manuals folder.

You have tools to:
- inspect directory structure (Tree/ListFiles)
- search across manuals (Search)
- read local evidence (ReadTextFile/ReadPdfPages)

Rules:
1) Prefer a file-first workflow: Tree/ListFiles -> Search -> Read evidence -> answer.
2) Use `Search.extensions` to keep searches targeted (e.g. start with `['pdf']` for manuals, or `['xlsx']` for QA sheets). Broaden only if needed.
3) Do multiple searches with synonyms/abbreviations if needed.
4) Never invent facts. If the manuals don't contain the answer, say "Not found in the provided manuals".
5) Always cite sources (file path + page/snippet/location) for factual claims.
6) Keep tool outputs small; request only the pages/snippets you need.
"""


def answer_question(
    *,
    model: str,
    docs_root: str,
    question: str,
    max_steps: int = 12,
    log_path: str | None = None,
) -> dict[str, Any]:
    api_key = _load_api_key()
    client = OpenAI(api_key=api_key)

    log: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {
            "role": "user",
            "content": f"Docs root: {docs_root}\n\nQuestion: {question}\n",
        },
    ]

    tool_state: dict[str, Any] = {"docs_root": docs_root}
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

    for i in range(max_steps):
        step_id = f"step_{i + 1}"
        started = time.time()

        completion = client.beta.chat.completions.parse(
            model=model,
            response_format=NextStep,
            messages=log,
            max_completion_tokens=2048,
        )

        msg = completion.choices[0].message
        job: NextStep = msg.parsed
        completion_text = _extract_completion_text(msg)

        if log_path:
            _write_jsonl(
                log_path,
                {
                    "event": "llm",
                    "step": step_id,
                    "duration_sec": time.time() - started,
                    "completion_text": completion_text,
                    "parsed": job.model_dump(),
                },
            )

        if isinstance(job.function, FinalizeAnswer):
            result = {
                "answer": job.function.answer,
                "citations": job.function.citations,
            }
            if log_path:
                _write_jsonl(log_path, {"event": "final", "result": result})
            return result

        log.append(
            {
                "role": "assistant",
                "content": job.plan_remaining_steps_brief[0],
                "tool_calls": [
                    {
                        "type": "function",
                        "id": step_id,
                        "function": {
                            "name": job.function.__class__.__name__,
                            "arguments": job.function.model_dump_json(),
                        },
                    }
                ],
            }
        )

        tool_result: ToolResult = dispatch_tool(job.function, tool_state)
        tool_content = tool_result.to_tool_content()

        if log_path:
            _write_jsonl(
                log_path,
                {
                    "event": "tool",
                    "step": step_id,
                    "tool": job.function.__class__.__name__,
                    "ok": tool_result.ok,
                    "content": tool_result.payload,
                },
            )

        log.append({"role": "tool", "content": tool_content, "tool_call_id": step_id})

    return {
        "answer": "Not found in the provided manuals (step limit reached).",
        "citations": [],
    }
