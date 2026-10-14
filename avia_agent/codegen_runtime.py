import bisect
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

_PDF_MARKER_CACHE: dict[str, list[tuple[int, int]]] = {}


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def find_ugrep_exe() -> str | None:
    override = (os.environ.get("AVIA_UGREP_PATH") or "").strip()
    if override and os.path.exists(override):
        return override

    local = os.path.join(_project_root(), "tools", "ugrep", "ugrep.exe")
    if os.path.exists(local):
        return local

    return shutil.which("ugrep")


def _normalize_exts(exts: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for e in exts or []:
        e = (e or "").strip().lower()
        if not e:
            continue
        if e.startswith("."):
            e = e[1:]
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def parse_ugrep_lines(out: str, max_results: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for line in (out or "").splitlines():
        if len(results) >= max_results:
            break
        line = line.strip("\r\n")
        if not line:
            continue

        m = re.match(r"^(?P<path>.*):(?P<line>\d+):(?P<rest>.*)$", line)
        if not m:
            continue
        path = m.group("path").strip()
        line_no = m.group("line").strip()
        rest = m.group("rest").strip()
        try:
            line_i = int(line_no)
        except ValueError:
            line_i = 0
        results.append(
            {
                "path": path,
                "location": f"line {line_no}",
                "line": line_i,
                "snippet": rest,
            }
        )
    return results


def _run_one(
    *,
    ugrep: str,
    cwd: str,
    env: dict[str, str],
    query: str,
    mode: str,
    max_results: int,
    filter_arg: str | None,
    only_exts: str | None,
    root: str,
) -> tuple[int, str, str]:
    cmd: list[str] = [
        ugrep,
        "--color=never",
        "-r",
        "-n",
        "-H",
        "-i",
        "--no-messages",
        "-m",
        str(max_results),
    ]
    if filter_arg:
        cmd.append(filter_arg)
    if only_exts:
        cmd.append("-O" + only_exts)
    if mode == "fixed":
        cmd.append("-F")
    cmd.extend([query, root])

    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=False,
        cwd=cwd,
        env=env,
    )
    return p.returncode, (p.stdout or ""), (p.stderr or "")


def _pdf_marker_map(
    *, ugrep: str, cwd: str, env: dict[str, str], py: str, pdf_path: str
) -> list[tuple[int, int]]:
    cached = _PDF_MARKER_CACHE.get(pdf_path)
    if cached is not None:
        return cached

    cmd: list[str] = [
        ugrep,
        "--color=never",
        "-n",
        "-H",
        "--no-messages",
        "--filter=pdf:" + py + " -m avia_agent.filter_stdin pdf",
        "-Opdf",
        "-F",
        "@@@PAGE",
        pdf_path,
    ]
    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=False,
        cwd=cwd,
        env=env,
    )
    markers: list[tuple[int, int]] = []
    for raw in (p.stdout or "").splitlines():
        m = re.match(r"^(?P<path>.*):(?P<line>\d+):(?P<rest>.*)$", raw.strip())
        if not m:
            continue
        try:
            line_i = int(m.group("line"))
        except ValueError:
            continue
        rest = m.group("rest")
        m2 = re.search(r"@@@PAGE\s+(\d+)@@@", rest, flags=re.IGNORECASE)
        if not m2:
            continue
        try:
            page_i = int(m2.group(1))
        except ValueError:
            continue
        markers.append((line_i, page_i))
    markers.sort()
    _PDF_MARKER_CACHE[pdf_path] = markers
    return markers


def _resolve_pdf_pages_for_results(
    *, ugrep: str, cwd: str, env: dict[str, str], py: str, results: list[dict[str, Any]]
) -> None:
    by_pdf: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        path = (r.get("path") or "").strip()
        if path.lower().endswith(".pdf"):
            by_pdf.setdefault(path, []).append(r)

    for pdf_path, hits in by_pdf.items():
        markers = _pdf_marker_map(ugrep=ugrep, cwd=cwd, env=env, py=py, pdf_path=pdf_path)
        if not markers:
            continue
        marker_lines = [m[0] for m in markers]
        marker_pages = [m[1] for m in markers]
        for h in hits:
            line_i = int(h.get("line") or 0)
            idx = bisect.bisect_right(marker_lines, line_i) - 1
            if idx >= 0:
                page = marker_pages[idx]
                h["page"] = page
                h["location"] = f"page {page}"


def ugrep_search(
    *,
    docs_root: str,
    query: str,
    mode: str = "fixed",
    extensions: Sequence[str] = ("pdf",),
    max_results: int = 30,
) -> dict[str, Any]:
    ugrep = find_ugrep_exe()
    if not ugrep:
        return {
            "ok": False,
            "error": "ugrep not found (run scripts/bootstrap.ps1 or set AVIA_UGREP_PATH)",
        }

    docs_root = os.path.abspath(docs_root)
    project_root = _project_root()
    py = sys.executable

    wanted = set(_normalize_exts(extensions))
    if not wanted:
        wanted = {"pdf"}

    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    outs: list[str] = []
    errs: list[str] = []
    passes: list[dict[str, Any]] = []
    ok = True

    def run_pass(filter_value: str | None, only_exts: str | None):
        nonlocal ok
        started = time.time()
        rc, out, err = _run_one(
            ugrep=ugrep,
            cwd=project_root,
            env=env,
            query=query,
            mode=mode,
            max_results=max_results,
            filter_arg=filter_value,
            only_exts=only_exts,
            root=docs_root,
        )
        passes.append(
            {
                "only_exts": only_exts or "",
                "filter": filter_value or "",
                "returncode": rc,
                "duration_sec": time.time() - started,
                "stdout_lines": len((out or "").splitlines()),
                "stderr_lines": len((err or "").splitlines()),
            }
        )
        if rc not in (0, 1):
            ok = False
        if out.strip():
            outs.append(out.strip())
        if err.strip():
            errs.append(err.strip())

    if "pdf" in wanted:
        run_pass(f"--filter=pdf:{py} -m avia_agent.filter_stdin pdf", "pdf")
    if "docx" in wanted:
        run_pass(f"--filter=docx:{py} -m avia_agent.filter_stdin docx", "docx")
    if "pptx" in wanted:
        run_pass(f"--filter=pptx:{py} -m avia_agent.filter_stdin pptx", "pptx")
    if wanted.intersection({"xlsx", "xlsm", "xltx", "xltm"}):
        run_pass(
            f"--filter=xlsx,xlsm,xltx,xltm:{py} -m avia_agent.filter_stdin xlsx",
            "xlsx,xlsm,xltx,xltm",
        )
    if wanted.intersection({"txt", "md"}):
        run_pass(None, "txt,md")

    raw = "\n".join(outs)
    parsed = parse_ugrep_lines(raw, max_results=max_results)
    _resolve_pdf_pages_for_results(ugrep=ugrep, cwd=project_root, env=env, py=py, results=parsed)

    payload: dict[str, Any] = {
        "ok": ok,
        "ugrep": ugrep,
        "docs_root": docs_root,
        "query": query,
        "mode": mode,
        "extensions": list(extensions),
        "passes": passes,
        "results": parsed,
        "matches": parsed,
        "raw": raw,
    }
    if errs:
        payload["stderr"] = "\n".join(errs)
    return payload


def read_pdf_pages(path: str, page_start: int, page_end: int) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    n = len(reader.pages)
    page_start = max(1, min(page_start, n))
    page_end = max(1, min(page_end, n))
    if page_end < page_start:
        page_start, page_end = page_end, page_start

    parts: list[str] = []
    for i in range(page_start - 1, page_end):
        txt = reader.pages[i].extract_text() or ""
        parts.append(f"\n--- page {i + 1} ---\n{txt}")
    return "\n".join(parts)


@dataclass(frozen=True)
class QaRow:
    question: str
    level: str
    qtype: str
    labeled_answer: str
    labeled_pages: str
    sentences: str

    def to_tsv(self) -> str:
        sentences = self.sentences or ""
        return (
            f"{self.question}\t{self.level}\t{self.qtype}\t{self.labeled_answer}\t{self.labeled_pages}\t"
            + f'"{sentences}"'
        )


def lookup_xlsx_qa_row(path: str, question: str) -> QaRow | None:
    import openpyxl

    want = " ".join((question or "").strip().lower().split())
    if not want:
        return None

    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    for sheet in wb.worksheets:
        first = True
        for row in sheet.iter_rows(values_only=True):
            if first:
                first = False
                continue
            if not row or not row[0]:
                continue
            q = str(row[0]).strip()
            if " ".join(q.lower().split()) != want:
                continue

            def norm(v: Any) -> str:
                if v is None:
                    return ""
                if isinstance(v, float) and v.is_integer():
                    return str(int(v))
                return str(v).strip()

            return QaRow(
                question=q,
                level=norm(row[1] if len(row) > 1 else ""),
                qtype=norm(row[2] if len(row) > 2 else ""),
                labeled_answer=norm(row[3] if len(row) > 3 else ""),
                labeled_pages=norm(row[4] if len(row) > 4 else ""),
                sentences=norm(row[5] if len(row) > 5 else ""),
            )
    return None
