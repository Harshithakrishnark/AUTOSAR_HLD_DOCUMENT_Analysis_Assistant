# AUTOSAR HLD Document Analysis Assistant

A starter implementation aligned with Case Study 1 in the supplied Automotive Engineering AI case-study PDF.

## Features
- Upload PDF high-level design (HLD) documents
- Extract text with page numbers
- Search extracted text and return page-referenced snippets
- Ask questions using evidence retrieved from the document
- Optional local LLM integration through Ollama
- Baseline ML classifier for labeling automotive document sections
- Basic evaluation script for the classifier

> This is a pilot/student project, not a certified automotive engineering tool. Validate all outputs with a qualified engineer. Scanned PDFs may need OCR, which is not included in this starter version.

## Requirements
- Windows 10/11
- Python 3.10+ (3.11 recommended)
- Optional: Ollama for local LLM answers

## Setup (Windows PowerShell)
1. Extract the ZIP.
2. Open the extracted `autosar_hld_document_analysis_assistant` folder in VS Code.
3. Open Terminal > New Terminal.
4. Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this in the same terminal, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

5. Install packages:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

6. Run the app:

```powershell
streamlit run app/main.py
```

## Optional local LLM
Install Ollama from https://ollama.com/download/windows, then in a terminal run:

```powershell
ollama pull llama3.2:3b
```

Keep Ollama running. In another terminal, run the Streamlit app. If Ollama is not available, the app still works in retrieval-only mode and clearly says it has not generated an LLM answer.

## ML dataset and evaluation
The project includes `data/sample_sections.csv`, a small illustrative seed dataset. It is not a real automotive benchmark and is too small for claims about generalization. Replace or expand it with reviewed, appropriately licensed data before reporting meaningful results.

Run:

```powershell
python -m app.train_model
```

This trains a baseline TF-IDF + Logistic Regression classifier and saves it under `models/`. For a meaningful evaluation, add more labeled examples per class and use a held-out test set.

## Project structure
```text
autosar_hld_document_analysis_assistant/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── document_utils.py
│   ├── rag.py
│   └── train_model.py
├── data/
│   └── sample_sections.csv
├── models/
├── reports/
├── tests/
│   └── test_document_utils.py
├── requirements.txt
└── README.md
```
