import re
import sys

sys.path.insert(0, ".")
from avia_agent.extract_text import extract_text

PDF_FILE = "OM A XYZ Aviation-1.pdf"

if len(sys.argv) < 2:
    print("Usage: py read_pdf.py PAGE [END_PAGE]")
    sys.exit(1)

start = int(sys.argv[1])
end = int(sys.argv[2]) if len(sys.argv) > 2 else start


text = extract_text(PDF_FILE)
pages = re.split(r"\n@@@PAGE (\d+)@@@\n", text)


page_dict = {}
for i in range(1, len(pages), 2):
    page_dict[int(pages[i])] = pages[i + 1] if i + 1 < len(pages) else ""


for num in range(start, end + 1):
    if num in page_dict:
        print(f"=== PAGE {num} ===")
        print(page_dict[num])
