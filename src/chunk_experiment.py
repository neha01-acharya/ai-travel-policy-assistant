from ingestion import load_documents, chunk_text


def run_experiment(documents, chunk_size, chunk_overlap):
    total_chunks = 0
    chunks_per_document = {}

    for document in documents:
        chunks = chunk_text(
            document["content"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        total_chunks += len(chunks)
        chunks_per_document[document["source"]] = len(chunks)

    return total_chunks, chunks_per_document


if __name__ == "__main__":
    documents = load_documents()

    experiments = [
        (300, 50),
        (500, 100),
        (800, 150),
    ]

    print("\nCHUNKING EXPERIMENT")
    print("=" * 60)

    for chunk_size, chunk_overlap in experiments:

        total_chunks, chunks_per_document = run_experiment(
            documents,
            chunk_size,
            chunk_overlap
        )

        print(f"\nChunk size : {chunk_size}")
        print(f"Overlap    : {chunk_overlap}")
        print(f"Total chunks: {total_chunks}")

        print("\nChunks per document:")

        for source, count in chunks_per_document.items():
            print(f"  {source}: {count}")

        print("-" * 60)