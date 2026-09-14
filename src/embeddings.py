import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunks.json"

VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"

EMBEDDINGS_FILE = VECTOR_STORE_DIR / "embeddings.npy"

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():
    """Load processed policy chunks from JSON."""

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_FILE}"
        )

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        raise ValueError("chunks.json is empty.")

    return chunks


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_embedding_model():
    """Load the Sentence Transformer model."""

    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded.")

    return model


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(chunks, model):
    """Convert policy chunk text into embedding vectors."""

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

def save_embeddings(embeddings):
    """Save embedding vectors as a NumPy file."""

    # Create vector_store directory if it doesn't exist
    VECTOR_STORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save embeddings
    np.save(
        EMBEDDINGS_FILE,
        embeddings
    )

    print(
        f"\nEmbeddings saved to: {EMBEDDINGS_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("POLICY EMBEDDING GENERATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Step 1: Load chunks
    # --------------------------------------------------------

    chunks = load_chunks()

    print(
        f"\nLoaded chunks: {len(chunks)}"
    )

    # --------------------------------------------------------
    # Step 2: Load embedding model
    # --------------------------------------------------------

    model = load_embedding_model()

    # --------------------------------------------------------
    # Step 3: Create embeddings
    # --------------------------------------------------------

    embeddings = create_embeddings(
        chunks,
        model
    )

    # --------------------------------------------------------
    # Step 4: Save embeddings
    # --------------------------------------------------------

    save_embeddings(embeddings)

    # --------------------------------------------------------
    # Step 5: Display information
    # --------------------------------------------------------

    print(
        f"\nEmbedding shape: {embeddings.shape}"
    )

    print(
        f"Number of embeddings: {len(embeddings)}"
    )

    print(
        f"Embedding dimension: {embeddings.shape[1]}"
    )

    print("\nFirst embedding:")
    print(
        embeddings[0][:10]
    )

    print("\nEmbedding generation successful.")