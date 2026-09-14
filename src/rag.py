import json
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunks.json"

EMBEDDINGS_FILE = (
    BASE_DIR / "data" / "vector_store" / "embeddings.npy"
)

INDEX_FILE = (
    BASE_DIR / "data" / "vector_store" / "policy.index"
)

MODEL_NAME = "all-MiniLM-L6-v2"

LLM_MODEL = "gemini-3.6-flash"

TOP_K = 3

SIMILARITY_THRESHOLD = 0.40


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():
    """Load policy chunks and metadata from JSON."""

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_FILE}"
        )

    with CHUNKS_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:
        chunks = json.load(file)

    if not chunks:
        raise ValueError(
            "chunks.json is empty."
        )

    return chunks


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings():
    """Load previously generated embeddings."""

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {EMBEDDINGS_FILE}"
        )

    embeddings = np.load(
        EMBEDDINGS_FILE
    )

    if embeddings.size == 0:
        raise ValueError(
            "Embeddings file is empty."
        )

    return embeddings.astype(
        "float32"
    )


# ============================================================
# LOAD FAISS INDEX
# ============================================================

def load_faiss_index():
    """Load the saved FAISS vector index."""

    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_FILE}"
        )

    index = faiss.read_index(
        str(INDEX_FILE)
    )

    return index


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_embedding_model():
    """Load the same embedding model used during indexing."""

    print(
        f"Loading embedding model: {MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        "Embedding model loaded."
    )

    return model


# ============================================================
# LOAD GEMINI
# ============================================================

def load_llm():
    """Load Gemini for RAG answer generation."""

    print(
        f"Loading LLM: {LLM_MODEL}"
    )

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0
    )

    print(
        "LLM loaded."
    )

    return llm


# ============================================================
# SEARCH POLICY
# ============================================================

def search_policy(
    query,
    model,
    index,
    chunks,
    top_k=TOP_K
):
    """
    Convert the user question into an embedding
    and retrieve the most relevant policy chunks.
    """

    if not query or not query.strip():
        return []

    # --------------------------------------------------------
    # Convert query into an embedding
    # --------------------------------------------------------

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = query_embedding.astype(
        "float32"
    )

    # --------------------------------------------------------
    # Search FAISS
    # --------------------------------------------------------

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0]
    ):

        # FAISS uses -1 when no result exists
        if index_position == -1:
            continue

        # Ignore results below similarity threshold
        if score < SIMILARITY_THRESHOLD:
            continue

        chunk = chunks[index_position]

        results.append(
            {
                "score": float(score),
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": chunk["metadata"]
            }
        )

    return results


# ============================================================
# BUILD POLICY CONTEXT
# ============================================================

def build_context(results):
    """
    Convert retrieved chunks into a context block
    that will be provided to Gemini.
    """

    if not results:
        return ""

    context_parts = []

    for position, result in enumerate(
        results,
        start=1
    ):

        metadata = result["metadata"]

        source = metadata.get(
            "source",
            "Unknown"
        )

        policy_type = metadata.get(
            "policy_type",
            "Unknown"
        )

        country = metadata.get(
            "country",
            "Unknown"
        )

        context_parts.append(
            f"""
POLICY CHUNK {position}
Source: {source}
Policy Type: {policy_type}
Country: {country}
Similarity Score: {result['score']:.4f}

{result['text']}
""".strip()
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# GENERATE RAG ANSWER
# ============================================================

def generate_answer(
    query,
    results,
    llm
):
    """
    Generate a grounded answer using only
    the retrieved policy context.
    """

    if not results:
        return (
            "I couldn't find relevant information "
            "in the company travel policies to answer "
            "that question."
        )

    context = build_context(
        results
    )

    prompt = f"""
You are an AI assistant for a company's
corporate travel policy.

Your job is to answer the employee's question
using ONLY the policy context provided below.

STRICT RULES:

1. Use only the provided policy context.
2. Do not invent policies, limits, fees, approvals,
   exceptions, or reimbursement amounts.
3. If the answer is not available in the context,
   clearly say that the policy information is not available.
4. Give a concise and direct answer.
5. If a spending limit or amount is mentioned,
   preserve the correct currency.
6. If approval is required, clearly explain that.
7. Do not claim that an approval exists unless the
   context explicitly provides such evidence.
8. Mention the relevant policy source at the end.

EMPLOYEE QUESTION:
{query}

POLICY CONTEXT:
{context}

ANSWER:
"""

    response = llm.invoke(
        prompt
    )

    # --------------------------------------------------------
    # Handle different Gemini response formats
    # --------------------------------------------------------

    content = response.content

    if isinstance(content, str):
        answer = content

    elif isinstance(content, list):
        text_parts = []

        for item in content:

            if isinstance(item, str):
                text_parts.append(item)

            elif isinstance(item, dict):

                text = item.get(
                    "text"
                )

                if text:
                    text_parts.append(text)

        answer = "\n".join(
            text_parts
        )

    else:
        answer = str(content)

    return answer.strip()


# ============================================================
# COMPLETE RAG PIPELINE
# ============================================================

def ask_policy(
    query,
    model,
    index,
    chunks,
    llm,
    top_k=TOP_K
):
    """
    Complete RAG pipeline:

    Question
        ↓
    Query embedding
        ↓
    FAISS search
        ↓
    Relevant chunks
        ↓
    Context
        ↓
    Gemini
        ↓
    Answer + sources
    """

    results = search_policy(
        query=query,
        model=model,
        index=index,
        chunks=chunks,
        top_k=top_k
    )

    answer = generate_answer(
        query=query,
        results=results,
        llm=llm
    )

    return {
        "answer": answer,
        "sources": results
    }


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_response(
    query,
    response
):
    """Display the final RAG response."""

    print("\n" + "=" * 60)
    print("RAG RESPONSE")
    print("=" * 60)

    print(
        f"\nQuestion:\n{query}"
    )

    print(
        f"\nAnswer:\n{response['answer']}"
    )

    print(
        "\nSources:"
    )

    if not response["sources"]:
        print(
            "No policy sources found."
        )
        return

    seen_sources = set()

    for source in response["sources"]:

        source_name = source["metadata"].get(
            "source",
            "Unknown"
        )

        if source_name in seen_sources:
            continue

        seen_sources.add(
            source_name
        )

        print(
            f"- {source_name}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AI TRAVEL POLICY RAG")
    print("=" * 60)

    # --------------------------------------------------------
    # Load policy chunks
    # --------------------------------------------------------

    chunks = load_chunks()

    print(
        f"\nLoaded chunks: {len(chunks)}"
    )

    # --------------------------------------------------------
    # Load embeddings
    # --------------------------------------------------------

    embeddings = load_embeddings()

    print(
        f"Loaded embeddings: {embeddings.shape}"
    )

    # --------------------------------------------------------
    # Load FAISS index
    # --------------------------------------------------------

    index = load_faiss_index()

    print(
        f"Loaded FAISS index."
    )

    print(
        f"Vectors in index: {index.ntotal}"
    )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    model = load_embedding_model()

    # --------------------------------------------------------
    # Load Gemini
    # --------------------------------------------------------

    llm = load_llm()

    # --------------------------------------------------------
    # Test questions
    # --------------------------------------------------------

    test_queries = [
        "What is the airport reimbursement limit in India?",
        "Can I take an airport trip at night?",
        "What happens if my trip costs more than the limit?",
        "Is hotel reimbursement covered by the travel policy?"
    ]

    # --------------------------------------------------------
    # Run RAG
    # --------------------------------------------------------

    for query in test_queries:

        response = ask_policy(
            query=query,
            model=model,
            index=index,
            chunks=chunks,
            llm=llm,
            top_k=TOP_K
        )

        display_response(
            query=query,
            response=response
        )

    print(
        "\n" + "=" * 60
    )

    print(
        "RAG TEST COMPLETED"
    )

    print(
        "=" * 60
    )