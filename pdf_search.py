import re
import sys

from pypdf import PdfReader

PDF_FILE = "OM A XYZ Aviation-1.pdf"


def search_pdf(query: str, pdf_path: str = PDF_FILE):
    reader = PdfReader(pdf_path)
    pattern = re.compile(re.escape(query), re.IGNORECASE)

    results = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        for match in pattern.finditer(text):
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text[start:end].replace("\n", " ").strip()

            results.append(f"[{page_num}] ...{context}...")

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: py pdf_search.py 'query' [pdf_file]")
        sys.exit(1)

    query = sys.argv[1]
    pdf_path = sys.argv[2] if len(sys.argv) > 2 else PDF_FILE

    results = search_pdf(query, pdf_path)

    if not results:
        print(f"No matches for '{query}'")
        sys.exit(1)

    for r in results:
        print(r)
        print()


if __name__ == "__main__":
    main()
