from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import sys

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF_DIR = PROJECT_ROOT / "data" / "pdf"
DEFAULT_VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"
DEFAULT_COLLECTION_NAME = "Pdf_documents"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _create_vector_store(
    store_dir: Path, collection_name: str, embedding_model: str
) -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
    )
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(store_dir),
        embedding_function=embeddings,
    )


def _chunk_id(source: str, page: int, start_index: int) -> str:
    return sha256(f"{Path(source).name}:{page}:{start_index}".encode("utf-8")).hexdigest()


def _ingest(args: argparse.Namespace) -> None:
    pdf_files = sorted(args.pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {args.pdf_dir}")

    documents = []
    for pdf_path in pdf_files:
        page_documents = PyPDFLoader(str(pdf_path)).load()
        documents.extend(page_documents)
        print(f"Loaded {len(page_documents)} page(s) from {pdf_path.name}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError(f"No extractable text found in PDFs in {args.pdf_dir}")

    ids = [
        _chunk_id(
            chunk.metadata["source"],
            chunk.metadata.get("page", 0),
            chunk.metadata.get("start_index", 0),
        )
        for chunk in chunks
    ]
    vector_store = _create_vector_store(
        args.store_dir, args.collection_name, args.embedding_model
    )
    vector_store.add_documents(documents=chunks, ids=ids)

    print(
        f"Indexed {len(chunks)} chunk(s) from {len(pdf_files)} PDF(s) "
        f"({len(documents)} page(s)) in collection {args.collection_name!r}."
    )
    print(f"Persistent store: {args.store_dir}")


def _search(args: argparse.Namespace) -> None:
    vector_store = _create_vector_store(
        args.store_dir, args.collection_name, args.embedding_model
    )
    results = vector_store.similarity_search_with_score(args.query, k=args.limit)
    if not results:
        print("No indexed PDF chunks found. Run `rag ingest` first.")
        return

    for rank, (document, distance) in enumerate(results, start=1):
        source = Path(document.metadata["source"]).name
        page = document.metadata.get("page")
        page_label = f"page {page + 1}" if isinstance(page, int) else "page unknown"
        print(f"[{rank}] {source}, {page_label} (distance: {distance:.4f})")
        print(document.page_content)
        print()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Index local PDF files and search their persistent vector store.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ingest_parser = commands.add_parser("ingest", help="Index PDFs from a directory.")
    ingest_parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    ingest_parser.add_argument(
        "--store-dir", type=Path, default=DEFAULT_VECTOR_STORE_DIR
    )
    ingest_parser.add_argument(
        "--collection-name", default=DEFAULT_COLLECTION_NAME
    )
    ingest_parser.add_argument(
        "--embedding-model", default=DEFAULT_EMBEDDING_MODEL
    )
    ingest_parser.set_defaults(handler=_ingest)

    search_parser = commands.add_parser("search", help="Search indexed PDF chunks.")
    search_parser.add_argument("query", help="Question or phrase to search for.")
    search_parser.add_argument(
        "-k", "--limit", type=int, default=4, help="Number of results (default: 4)."
    )
    search_parser.add_argument(
        "--store-dir", type=Path, default=DEFAULT_VECTOR_STORE_DIR
    )
    search_parser.add_argument(
        "--collection-name", default=DEFAULT_COLLECTION_NAME
    )
    search_parser.add_argument(
        "--embedding-model", default=DEFAULT_EMBEDDING_MODEL
    )
    search_parser.set_defaults(handler=_search)
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")

    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "search" and args.limit < 1:
        parser.error("--limit must be greater than zero")
    args.handler(args)
