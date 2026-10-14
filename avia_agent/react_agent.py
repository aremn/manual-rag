import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from .codegen_runtime import read_pdf_pages, ugrep_search

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ugrep_search",
            "description": "Search for text in documents (PDF, DOCX, XLSX). Returns list of matches with file path, page/location, and snippet. Use SHORT search terms (1-3 words) for best results. Search for key concepts, not full phrases.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query - use short key terms (1-3 words), not long phrases. Examples: 'fuel policy', 'MEL', 'inexperienced pilot', 'bird strike'",
                    },
                    "extensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "File extensions to search. Default: ['pdf']. Options: 'pdf', 'docx', 'xlsx'",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_pdf_pages",
            "description": "Read specific pages from a PDF file. Use after ugrep_search to read full content of relevant pages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to PDF file (from ugrep_search results)",
                    },
                    "start_page": {
                        "type": "integer",
                        "description": "First page to read (1-indexed)",
                    },
                    "end_page": {
                        "type": "integer",
                        "description": "Last page to read (1-indexed). Keep range small (1-3 pages).",
                    },
                },
                "required": ["path", "start_page", "end_page"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": "Provide the final answer to the user's question. Call this when you have gathered enough information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The answer to the question, based ONLY on information found in the documents. If not found, say 'Not found in the provided manuals'.",
                    },
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "page": {"type": "integer"},
                                "quote": {
                                    "type": "string",
                                    "description": "Relevant quote from the page (1-2 sentences)",
                                },
                            },
                            "required": ["path", "page"],
                        },
                        "description": "List of citations supporting the answer. Include page numbers.",
                    },
                },
                "required": ["answer", "citations"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are an aviation documentation assistant. Answer questions using ONLY information from the provided manuals.

WORKFLOW:
1. THINK OUT LOUD - before each action, explain your reasoning in 1-2 sentences
2. ANALYZE the question to identify KEY CONCEPTS (not just keywords)
3. Search using ugrep_search with SHORT terms (1-3 words)
4. Read relevant pages with read_pdf_pages
5. If answer is incomplete, search for RELATED CONCEPTS
6. Call answer() with findings and citations

IMPORTANT: Always explain your thinking before calling a tool. Example:
"The question asks about inexperienced pilots pairing. I'll search for 'inexperienced' first."
Then call the tool.

SEARCH STRATEGY - Think about what the manual would contain:

1. For "X is inoperative/failed/unserviceable" questions:
   - Search for: "failed or downgraded", "downgraded equipment", "effect on minima"
   - The answer is usually in TABLES about equipment failures
   - Also search for X itself to understand context

2. For "how does person X do Y" or "X's responsibilities":
   - First search for general rules about Y (may apply to all managers/roles)
   - Then search for X-specific information
   - Rules about delegation, authority, etc. are often in GENERAL sections

3. For definitions ("what is X", "what defines X"):
   - Search for "X definition" or just "X"
   - Look in glossary sections or chapter beginnings

4. For procedures ("how to", "when to", "what happens if"):
   - Search for the action/event name
   - Search for related procedures or checklists

5. For documents/manuals (Training Manual, Operations Manual, etc.):
   - Aviation manuals use CODES like "OM-A", "OM-D", "Part D"
   - "Training Manual" might be called "OM-D" or "Part D, Training"
   - Search for both the full name AND the code/abbreviation

SEARCH TIPS:
- Use SHORT terms: "fuel policy", "MEL", "bird strike", "delegation"
- Search for SYNONYMS: "inoperative" = "failed" = "downgraded" = "unserviceable"  
- Search for the CONCEPT, not exact question words
- Aviation manuals have TABLES for equipment failures - search "failed or downgraded"
- If few results, try partial words: "delegat" finds delegate/delegation/delegated
- Try alternate names: "Training Manual" = "OM-D" = "Part D"

IMPORTANT:
- Base your answer ONLY on what you find in the documents
- If content seems incomplete, try alternative search terms before giving up
- Always cite specific page numbers
- Keep quotes concise (1-2 sentences)
"""


def execute_tool(name: str, args: dict, docs_root: str, verbose: bool = False) -> Any:
    if name == "ugrep_search":
        query = args.get("query", "")
        extensions = tuple(args.get("extensions", ["pdf"]))
        results = ugrep_search(docs_root=docs_root, query=query, extensions=extensions)

        matches = results.get("matches", [])
        if verbose:
            print(f"[ugrep] query='{query}' results={len(matches)}")

        return matches[:20]

    elif name == "read_pdf_pages":
        path = args.get("path", "")
        start = args.get("start_page", 1)
        end = args.get("end_page", 1)

        if not os.path.isabs(path):
            path = os.path.join(docs_root, path)
        text = read_pdf_pages(path, start, end)
        if verbose:
            print(f"[pdf] read {path} pages {start}-{end}")
        return text

    elif name == "answer":
        return args

    else:
        return {"error": f"Unknown tool: {name}"}


def answer_with_react(
    question: str,
    docs_root: str,
    model: str = "gpt-5.1",
    verbose: bool = False,
    max_iterations: int = 10,
) -> dict:
    client = OpenAI()
    docs_root = str(Path(docs_root).resolve())

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Question: {question}\n\nDocuments are in: {docs_root}",
        },
    ]

    for iteration in range(max_iterations):
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"[ITERATION {iteration + 1}/{max_iterations}]")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        messages.append(msg)

        if verbose and msg.content:
            print(f"\n[THOUGHT] {msg.content}")

        if not msg.tool_calls:
            if verbose:
                print("[react] no tool calls, finishing")
            return {"answer": msg.content or "No answer generated", "citations": []}

        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            if verbose:
                print(f"\n[ACTION] {name}")
                print(f"  args: {json.dumps(args, ensure_ascii=False, indent=2)[:200]}")

            result = execute_tool(name, args, docs_root, verbose)

            if verbose and name != "answer":
                if isinstance(result, list):
                    print(f"\n[OBSERVATION] {len(result)} results found")
                    for r in result[:3]:
                        loc = r.get("location", "")
                        snip = r.get("snippet", "")[:80]
                        print(f"  - {loc}: {snip}...")
                elif isinstance(result, str):
                    print(f"\n[OBSERVATION] {len(result)} chars read")
                    print(f"  {result[:200]}...")

            if name == "answer":
                return {
                    "answer": result.get("answer", ""),
                    "citations": result.get("citations", []),
                }

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False)[:8000],
                }
            )

    return {"answer": "Could not find answer within iteration limit", "citations": []}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m avia_agent.react_agent <question>")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    result = answer_with_react(question, ".", verbose=True)

    print("\n" + "=" * 60)
    print("ANSWER:", result["answer"])
    print("\nCITATIONS:")
    for c in result["citations"]:
        print(f"  - {c.get('path')} page {c.get('page')}")
