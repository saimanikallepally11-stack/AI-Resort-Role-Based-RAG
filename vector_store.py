
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from rag_engine import load_role_documents, split_documents

BASE_DIR = Path(__file__).resolve().parent
VECTOR_DB_DIR = BASE_DIR / "vector_db"

load_dotenv(BASE_DIR / ".env")

ALLOWED_ROLES = {"guest", "front_desk", "operations"}


def get_embeddings():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("Gemini API key is missing from .env")

    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )


def build_vector_store():
    documents = load_role_documents()

    role_access = {
        "guest.txt": ["guest", "front_desk", "operations"],
        "front_desk.txt": ["front_desk", "operations"],
        "operations.txt": ["operations"]
    }

    role_documents = []

    for document in documents:
        source = document.metadata["source"]

        for role in role_access.get(source, []):
            role_documents.append(
                Document(
                    page_content=document.page_content,
                    metadata={"source": source, "role": role}
                )
            )

    chunks = split_documents(role_documents)

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=get_embeddings()
    )

    VECTOR_DB_DIR.mkdir(exist_ok=True)
    vector_store.save_local(str(VECTOR_DB_DIR))

    return vector_store


def load_vector_store():
    if not VECTOR_DB_DIR.exists():
        raise FileNotFoundError(
            "Vector database not found. Run python vector_store.py first."
        )

    return FAISS.load_local(
        str(VECTOR_DB_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True
    )


def search_role_documents(question, role):
    if role not in ALLOWED_ROLES:
        raise ValueError("Invalid user role.")

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    vector_store = load_vector_store()

    return vector_store.similarity_search(
        question,
        k=3,
        filter={"role": role},
        fetch_k=12
    )



if __name__ == "__main__":
    print("Testing role-based document permissions...")

    test_cases = [
        ("What is the check-in time?", "guest"),
        ("What is the front desk booking procedure?", "front_desk"),
        ("What is the historical guest experience score?", "operations")
    ]

    for question, role in test_cases:
        print(f"\nRole: {role}")
        print(f"Question: {question}")

        results = search_role_documents(question, role)

        for result in results:
            print("Source:", result.metadata["source"])
            print("Access role:", result.metadata["role"])

