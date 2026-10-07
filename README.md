# RAG

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed,
then open a terminal in the project directory and run:

```powershell
uv sync --link-mode=copy
```

The project requires Python 3.13 or newer (`uv sync` creates `.venv`).
Place PDF files in `data/pdf/`; the repository includes sample PDFs.

## Index PDFs and search

From the project directory, index all PDFs:

```powershell
uv run rag ingest
```

Then search the indexed content:

```powershell
uv run rag search "What is the main idea behind the Transformer architecture?"
```

The search command prints the four closest chunks by default. Use `-k 8` to
return more results. Both commands accept `--help` for options. Ingestion splits
PDF pages into overlapping chunks, embeds them with
`sentence-transformers/all-MiniLM-L6-v2`, and stores them in the persistent
Chroma collection `Pdf_documents` under `data/vector_store/`. Stable chunk IDs
make rerunning ingestion update existing chunks instead of duplicating them.
Search scores are distances; lower scores indicate closer matches. This project
retrieves document passages but does not generate answers with a language model.

The same ingestion and retrieval steps are also available in
`notebook/pdf_loader.ipynb`.

## Push to GitHub

After making your changes, review and commit them, then push the `main` branch:

```powershell
git status
git add .
git status
git commit -m "Describe your changes"
git push -u origin main
```

The vector database and local Python environments are ignored because they are
generated locally. To rebuild the database after cloning, add PDFs under
`data/pdf/` and run `uv run rag ingest`.