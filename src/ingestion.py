from pathlib import Path
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
POLICY_DIR = BASE_DIR / "data" / "company_policy"


# ============================================================
# METADATA
# ============================================================

def get_metadata(filename: str) -> dict:
    """
    Create metadata based on the policy filename.
    """

    metadata = {
        "source": filename,
        "policy_type": "general",
        "country": "All"
    }

    if filename == "travel_policy_india.txt":
        metadata["policy_type"] = "travel"
        metadata["country"] = "India"

    elif filename == "travel_policy_us.txt":
        metadata["policy_type"] = "travel"
        metadata["country"] = "United States"

    elif filename == "airport_policy.txt":
        metadata["policy_type"] = "airport"

    elif filename == "employee_eligibility.txt":
        metadata["policy_type"] = "eligibility"

    elif filename == "expense_policy.txt":
        metadata["policy_type"] = "expense"

    elif filename == "cancellation_policy.txt":
        metadata["policy_type"] = "cancellation"

    elif filename == "approval_policy.txt":
        metadata["policy_type"] = "approval"

    return metadata
def clean_text(text: str) -> str:
    """
    Clean policy text while preserving important policy information.
    """

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove trailing spaces from each line
    lines = [line.strip() for line in text.split("\n")]

    # Remove excessive blank lines
    cleaned_lines = []

    previous_blank = False

    for line in lines:

        if line == "":
            if not previous_blank:
                cleaned_lines.append(line)

            previous_blank = True

        else:
            cleaned_lines.append(line)
            previous_blank = False

    # Reconstruct the document
    text = "\n".join(cleaned_lines)

    # Remove unnecessary whitespace around the document
    return text.strip()

# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents() -> list:
    """
    Load all policy TXT files from the company_policy directory.
    """

    documents = []

    if not POLICY_DIR.exists():
        raise FileNotFoundError(
            f"Policy directory not found: {POLICY_DIR}"
        )

    txt_files = sorted(POLICY_DIR.glob("*.txt"))

    if not txt_files:
        print("No policy TXT files found.")
        return documents

    for file_path in txt_files:

        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as error:
            print(f"Could not read {file_path.name}: {error}")
            continue

        # Remove unnecessary surrounding whitespace
        content = clean_text(content)

        # Handle empty documents
        if not content:
            print(f"Skipping empty document: {file_path.name}")
            continue

        metadata = get_metadata(file_path.name)

        document = {
            "source": file_path.name,
            "content": content,
            "metadata": metadata
        }

        documents.append(document)

    return documents
def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> list:
    """
    Split policy text using recursive, structure-aware splitting.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    return splitter.split_text(text)

def create_chunks(documents: list) -> list:
    """
    Create chunks for all loaded documents while preserving metadata.
    """

    all_chunks = []

    for document in documents:

        chunks = chunk_text(
            document["content"],
            chunk_size=500,
            chunk_overlap=100
        )

        for index, chunk in enumerate(chunks, start=1):

            chunk_record = {
                "chunk_id": f"{Path(document['source']).stem}_{index:03d}",
                "text": chunk,
                "metadata": document["metadata"].copy()
            }

            all_chunks.append(chunk_record)

    return all_chunks

def save_chunks(chunks: list) -> None:
    """
    Save processed chunks to a JSON file.
    """

    output_dir = BASE_DIR / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "chunks.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)

    print(f"\nChunks saved to: {output_file}")

# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":

    documents = load_documents()

    print(f"\nLoaded {len(documents)} documents.\n")

    chunks = create_chunks(documents)

    print(f"Created {len(chunks)} chunks.\n")

    save_chunks(chunks)

    for chunk in chunks:

        print("=" * 60)
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Metadata: {chunk['metadata']}")
        print(f"Text:\n{chunk['text'][:300]}")
        print("=" * 60)

    print(f"\nTotal chunks: {len(chunks)}")