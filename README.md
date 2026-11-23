py -m venv .venv# DocuQuery — AI-Based Document Question Answering System

## Project purpose
This is a Python application for the assignment: AI-based Document Question Answering
without NLP or RAG. It creates a Knowledge Base from the document uploaded by the user
and answers questions using exact-term search and simple rule-based inference.

## What it does
1. Accepts PDF, DOCX, and TXT documents.
2. Extracts text from the uploaded file.
3. Splits the text into sentence-like facts.
4. Stores facts with IDs and creates an inverted index.
5. Extracts explicit `IF ... THEN ...` rules.
6. Applies a limited forward-chaining check when a rule condition is supported by
   a separate document fact.
7. Answers questions using the best matching source facts and supported conclusions.
8. Shows the extracted Knowledge Base and reasoning trace.

## What it deliberately does NOT use
- NLP libraries or pretrained language models
- LLM APIs
- Embeddings
- Vector databases
- Retrieval-Augmented Generation (RAG)

The system uses deterministic tokenization and exact-term matching. It does not
understand synonyms or paraphrases, so ask questions using terms present in the file.

## Requirements
- Python 3.10+
- VS Code
- Internet connection for initial package installation

## Run in VS Code (Windows)
1. Extract `DocuQuery_VSCode_Project.zip`.
2. Open VS Code.
3. Select **File → Open Folder** and choose the extracted `DocuQuery_VSCode_Project` folder.
4. Open **Terminal → New Terminal**.
5. Create a virtual environment:

   ```powershell
   py -m venv .venv
   ```

6. Activate it:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt terminal and run:

   ```cmd
   .venv\Scripts\activate.bat
   ```

7. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

8. Start the application:

   ```powershell
   python -m streamlit run app.py
   ```

9. Open the local URL printed in the terminal (usually `http://localhost:8501`).

## How to demonstrate
- Upload a document chosen by you (not a preloaded sample).
- Show the number of extracted facts and indexed terms.
- Ask a question whose answer appears in the uploaded document.
- Open "Show matching source facts" to demonstrate the evidence.
- Ask a question about something absent from the document and show the no-match response.
- If your document contains a rule in the form `If condition, then conclusion.` and a separate fact supports the condition, show the inference tab.

## Example for testing the rule engine
Use a text document with these separate statements:

- `A battery stores electrical energy.`
- `If a battery stores electrical energy, then it can supply power to a device.`

The rule is only inferred when the condition is supported by a different extracted fact. The rule does not prove itself.

## Evaluation
Use `evaluation_template.csv` to record at least 15 test questions:
- 10 answerable questions
- 3 questions that test explicit rule inference (where the document supports the rule conditions)
- 2 unanswerable questions

Suggested measures:
- Answer accuracy = correct answers / answerable questions × 100
- Coverage = questions with at least one retrieved fact / all questions × 100
- Unsupported response rate = questions answered despite no source support / unanswerable questions × 100

Do not claim evaluation scores until you have run the tests and filled the template.

## Limitations
- Exact matching only; synonyms and paraphrases are not understood.
- Sentence splitting is punctuation-based and may split abbreviations incorrectly.
- PDF extraction works for selectable text, not scanned image-only PDFs.
- Rule inference supports a deliberately small explicit IF–THEN pattern.
- It returns source-grounded fact sentences instead of generating unrestricted prose.
