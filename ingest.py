
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load API key from .env
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "API key not found. Add GOOGLE_API_KEY or GEMINI_API_KEY to .env"
    )

# Define which roles can access each document
documents_folder = BASE_DIR / "documents"

role_access = {
    "guest.txt": ["guest", "front_desk", "operations"],
    "front_desk.txt": ["front_desk", "operations"],
    "operations.txt": ["operations"],
}

all_documents = []

# Load documents and assign role permissions
for filename, roles in role_access.items():
    file_path = documents_folder / filename

    if not file_path.exists():
        print(f"Missing file: {filename}")
        continue

    content = file_path.read_text(encoding="utf-8").strip()

    if not content:
        print(f"Empty document: {filename}")
        continue

    for role in roles:
        all_documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": filename,
                    "role": role
                }
            )
        )

if not all_documents:
    raise ValueError("No documents found. Check your documents folder.")

# Split documents into smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)

chunks = splitter.split_documents(all_documents)

# Create Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key
)

# Build and save the FAISS vector database
vector_store = FAISS.from_documents(chunks, embeddings)

vector_db_folder = BASE_DIR / "vector_db"
vector_db_folder.mkdir(exist_ok=True)

vector_store.save_local(str(vector_db_folder))

print("RAG document ingestion completed!")
print(f"Documents loaded: {len(all_documents)}")
print(f"Text chunks created: {len(chunks)}")
print(f"Vector database saved in: {vector_db_folder}")
