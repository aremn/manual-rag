# Manual RAG

Question answering over local PDF, DOCX, PPTX, XLSX and text files. Answers include references to the retrieved document content.

Includes tool-calling, ReAct and Python code-generation variants, plus a JSONL evaluation runner.

## Run

Requires Python 3.11+ and `OPENAI_API_KEY` in the process environment.

```sh
python -m venv .venv
python -m pip install -r requirements.txt
python -m avia_agent react-answer --docs ./manuals --question "What does the manual say about fuel checks?"
```

Put documents you can use in `manuals/`. No client manuals or evaluation conversations are included.

```sh
python -m avia_agent answer --docs ./manuals --question "YOUR_QUESTION"
python -m avia_agent eval --docs ./manuals --input ./questions.jsonl
python -m unittest -v test_retrieval
```

Evaluation input contains one JSON object per line with `question` and `answer` fields. Predictions and logs are excluded from Git.

Search uses `ugrep` when available and falls back to Python extraction. Set `AVIA_SEARCH_BACKEND=python` to select Python explicitly. The code-generation variant executes generated Python and is intended for trusted experiments.
