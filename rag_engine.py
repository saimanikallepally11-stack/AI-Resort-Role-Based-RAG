
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"

ROLE_FILES = {
    "guest": "guest.txt",
    "front_desk": "front_desk.txt",
    "operations": "operations.txt",
}


def load_role_documents():
    documents = []

    for role, filename in ROLE_FILES.items():
        file_path = DOCUMENTS_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Knowledge file not found: {file_path}"
            )

        content = file_path.read_text(
            encoding="utf-8"
        ).strip()

        if not content:
            raise ValueError(
                f"Knowledge file is empty: {filename}"
            )

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "role": role,
                    "source": filename
                }
            )
        )

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )

    return splitter.split_documents(documents)


if __name__ == "__main__":
    documents = load_role_documents()
    chunks = split_documents(documents)

    print("ROLE-BASED RAG DOCUMENT TEST")
    print("--------------------------------")

    print(f"Documents loaded: {len(documents)}")
    print(f"Text chunks created: {len(chunks)}")

    for chunk in chunks:
        print(
            f"Role: {chunk.metadata['role']} | "
            f"Source: {chunk.metadata['source']}"
        )

    print("--------------------------------")
    print("Document loading and splitting successful!")
