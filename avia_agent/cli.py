import argparse
import json
import os
import sys
import time

from .agent import answer_question
from .codegen_agent import answer_with_codegen
from .eval import run_eval
from .react_agent import answer_with_react


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="avia-agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    answer = sub.add_parser("answer", help="Answer a single question from local docs.")
    answer.add_argument("--docs", required=True, help="Root folder with manuals (PDF/DOCX/etc).")
    answer.add_argument("--question", required=True, help="User question.")
    answer.add_argument(
        "--model",
        default=os.environ.get("AVIA_MODEL", "gpt-5.1"),
        help="OpenAI model ID.",
    )
    answer.add_argument("--max-steps", type=int, default=12, help="Max tool-using steps.")
    answer.add_argument("--log-dir", default="logs", help="Where to write JSONL logs.")

    cg = sub.add_parser(
        "codegen-answer",
        help="Answer by generating and executing Python code (uses ugrep --filter).",
    )
    cg.add_argument("--docs", required=True, help="Root folder with manuals.")
    cg.add_argument("--question", required=True, help="User question.")
    cg.add_argument(
        "--model",
        default=os.environ.get("AVIA_MODEL", "gpt-5.1"),
        help="OpenAI model ID.",
    )
    cg.add_argument("--retries", type=int, default=3, help="Max codegen retries.")
    cg.add_argument(
        "--verbose",
        action="store_true",
        help="Print codegen attempts, generated code, and fallbacks.",
    )
    cg.add_argument(
        "--show-code",
        action="store_true",
        help="Print generated Python code to the terminal.",
    )
    cg.add_argument(
        "--no-fallback", action="store_true", help="Disable non-LLM fallback retrieval."
    )
    cg.add_argument("--print-sources", action="store_true", help="Print sources after the answer.")
    cg.add_argument("--log-dir", default="logs", help="Where to write JSONL logs.")

    react = sub.add_parser(
        "react-answer",
        help="Answer using ReAct agent with direct tool calls (simpler than codegen).",
    )
    react.add_argument("--docs", required=True, help="Root folder with manuals.")
    react.add_argument("--question", required=True, help="User question.")
    react.add_argument(
        "--model",
        default=os.environ.get("AVIA_MODEL", "gpt-4o"),
        help="OpenAI model ID.",
    )
    react.add_argument("--max-iterations", type=int, default=10, help="Max agent iterations.")
    react.add_argument("--verbose", action="store_true", help="Print agent steps.")

    ev = sub.add_parser("eval", help="Run over a JSONL dataset (question/answer).")
    ev.add_argument("--docs", required=True, help="Root folder with manuals.")
    ev.add_argument("--input", required=True, help="JSONL with at least {question, answer}.")
    ev.add_argument(
        "--model",
        default=os.environ.get("AVIA_MODEL", "gpt-5.1"),
        help="OpenAI model ID.",
    )
    ev.add_argument("--max-steps", type=int, default=12, help="Max tool-using steps per question.")
    ev.add_argument("--limit", type=int, default=0, help="0 means all.")
    ev.add_argument(
        "--out",
        default="",
        help="Write predictions JSONL here (default: outputs/preds-<ts>.jsonl).",
    )
    ev.add_argument("--log-dir", default="logs", help="Where to write per-question agent logs.")

    args = parser.parse_args(argv)

    if args.cmd == "answer":
        os.makedirs(args.log_dir, exist_ok=True)
        run_id = time.strftime("%Y%m%d-%H%M%S")
        log_path = os.path.join(args.log_dir, f"answer-{run_id}.jsonl")

        result = answer_question(
            model=args.model,
            docs_root=args.docs,
            question=args.question,
            max_steps=args.max_steps,
            log_path=log_path,
        )

        print(result["answer"].strip())
        if result.get("citations"):
            print("\nSources:")
            for c in result["citations"]:
                loc = c.get("location")
                if loc:
                    print(f"- {c['path']} ({loc})")
                else:
                    print(f"- {c['path']}")

        sys.stdout.flush()
        return 0

    if args.cmd == "codegen-answer":
        os.makedirs(args.log_dir, exist_ok=True)
        run_id = time.strftime("%Y%m%d-%H%M%S")
        log_path = os.path.join(args.log_dir, f"codegen-answer-{run_id}.jsonl")
        result = answer_with_codegen(
            model=args.model,
            docs_root=args.docs,
            question=args.question,
            max_retries=args.retries,
            log_path=log_path,
            verbose=bool(args.verbose),
            show_code=bool(args.show_code),
            enable_fallback=not bool(args.no_fallback),
        )
        print((result.get("answer") or "").strip())
        if (args.print_sources or args.verbose) and result.get("citations"):
            print("\nSources:")
            for c in result["citations"]:
                loc = c.get("location")
                if loc:
                    print(f"- {c['path']} ({loc})")
                else:
                    print(f"- {c['path']}")
        return 0

    if args.cmd == "react-answer":
        result = answer_with_react(
            question=args.question,
            docs_root=args.docs,
            model=args.model,
            verbose=bool(args.verbose),
            max_iterations=args.max_iterations,
        )
        print((result.get("answer") or "").strip())
        if args.verbose and result.get("citations"):
            print("\nSources:")
            for c in result["citations"]:
                page = c.get("page", "?")
                print(f"- {c['path']} (page {page})")
        return 0

    if args.cmd == "eval":
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs("outputs", exist_ok=True)
        run_id = time.strftime("%Y%m%d-%H%M%S")
        out_path = args.out.strip() or os.path.join("outputs", f"preds-{run_id}.jsonl")

        summary = run_eval(
            docs_root=args.docs,
            input_path=args.input,
            out_path=out_path,
            log_dir=args.log_dir,
            model=args.model,
            max_steps=args.max_steps,
            limit=args.limit,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
