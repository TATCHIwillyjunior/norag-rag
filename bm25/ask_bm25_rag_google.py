import argparse
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval.bm25 import BM25Index, BM25Retriever, tokenize as _tokenize
from retrieval.chunks import (
    chunk_document as shared_chunk_document,
    load_chunks,
    load_corpus,
)

# This script is the same RAG pipeline as ask_rag_google.py, with ONE difference: the retrieval.
#
# ask_rag_google.py finds the chunks whose MEANING is closest to the question (dense retrieval:
# embeddings + vector search in ChromaDB).
# This script finds the chunks that contain the WORDS of the question, giving more weight to rare words
# (sparse, or lexical, retrieval with BM25).
#
# Everything else is unchanged: the document, the chunking, the prompt and the generation.
# There are no embeddings and no ChromaDB here: BM25 only needs to count words.
#
# See bm25_challenge_part2.md.


# for this to work, you need to set the GOOGLE_API_KEY environment variable
# with your API key from https://aistudio.google.com/api-keys

# Gemini model used for generation. Any model listed at https://ai.google.dev/gemini-api/docs/models
# (or returned by list_available_models() below) should work. Models get retired over time.
# Validated as working on 2026-10-19: gemini-2.5-flash-lite, gemini-2.5-flash, gemini-3-flash-preview, gemini-3.5-flash-lite.
# Free tier quotas are per model: if you get a 429 error, switching to another model is a quick workaround.
MODEL = "gemini-3.5-flash-lite"
STREAM_ON = True

# Use this function to list available models and their details
def list_available_models():
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY to list Gemini models.")
    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    print("Available models:")
    models = client.models.list()
    for m in models:
        print(f"  id: {m.id}")

# The document to index and search
DOCUMENT_FILE = Path(__file__).resolve().parents[1] / "docs" / "vellora-duo-espresso-machine.txt"

# Chunking settings: the same as in ask_rag_google.py
CHUNK_SIZE = 800
OVERLAP_PERCENT = 10
MIN_PARAGRAPH_LENGTH = 20


def tokenize(text):
    """Keep the original challenge import path while using shared tokenization."""
    return _tokenize(text)


# Chunking: exactly the same as in ask_rag_google.py
def chunk_document(filepath):
    return shared_chunk_document(
        filepath,
        chunk_size=CHUNK_SIZE,
        overlap_percent=OVERLAP_PERCENT,
        min_paragraph_length=MIN_PARAGRAPH_LENGTH,
    )


# get_rag_context(index, query, n_results=5) performs a BM25 search
# to retrieve the most relevant snippets based on the query.
def get_rag_context(index, query, n_results=5):
    results = index.search(query, n_results)
    print("BM25 scores of the retrieved chunks: " + ", ".join(f"{score:.2f}" for _, score in results))
    return "\n---\n".join(chunk for chunk, _ in results)


def get_rag_results(index, query, n_results=5):
    """Return BM25 results using the shared retrieval-result shape."""
    return BM25Retriever(index).search(query, top_k=n_results)


# Generation Logic: exactly the same as in ask_rag_google.py
# Build a prompt that includes the retrieved context and the question,
# and send it to the model to generate a response.
def generate_response(question, context):
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY to generate an answer.")
    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    full_prompt = f"""
    You are a literary analyst. Using only the following excerpts from the book,
    answer the question provided. If the information is not in the excerpts, say you don't know.

    EXCERPTS:
    {context}

    QUESTION:
    {question}
    """

    if STREAM_ON:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end='', flush=True)
    else:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}]
        )
        print(response.choices[0].message.content)

# Execution
def main():
    parser = argparse.ArgumentParser(description="Run BM25 retrieval over a local document.")
    parser.add_argument(
        "--document",
        type=Path,
        default=DOCUMENT_FILE,
        help="UTF-8 document or corpus directory (defaults to the Vellora manual)",
    )
    parser.add_argument(
        "--question",
        default="What does error E47 mean and how do I fix it?",
        help="Question to retrieve evidence for",
    )
    parser.add_argument(
        "--corpus",
        help="Corpus name to use in results (defaults to the document or directory name)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Print retrieved evidence without calling Gemini",
    )
    args = parser.parse_args()
    document_file = args.document.resolve()
    if not document_file.is_file() and not document_file.is_dir():
        raise FileNotFoundError(f"document or corpus directory not found: {document_file}")

    chunks = (
        load_corpus(document_file, corpus=args.corpus)
        if document_file.is_dir()
        else load_chunks(
            document_file,
            chunk_size=CHUNK_SIZE,
            overlap_percent=OVERLAP_PERCENT,
            min_paragraph_length=MIN_PARAGRAPH_LENGTH,
            corpus=args.corpus,
        )
    )
    index = BM25Index(chunks)
    print(f"Built a BM25 index of {len(chunks)} chunks from {document_file} (in memory).")

    print(f"\n--- RAG RETRIEVAL START ---")
    results = get_rag_results(index, args.question, n_results=args.top_k)
    if not results:
        print("No chunks matched the query; no answer request was sent.")
        return
    print(
        "BM25 scores of the retrieved chunks: "
        + ", ".join(f"{result['score']:.2f}" for result in results)
    )
    context = "\n---\n".join(result["text"] for result in results)

    # Debug - print the retrieved context to verify it's working
    print(f"Retrieved context:\n{context}")

    if args.retrieval_only:
        return

    print("Retrieved relevant snippets. Sending to Gemini...")
    print("--- RESPONSE ---\n")
    generate_response(args.question, context)


if __name__ == "__main__":
    main()
