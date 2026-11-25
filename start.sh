
#!/bin/bash
set -e

echo "Building role-based FAISS vector database..."

python -c "from vector_store import build_vector_store; build_vector_store()"

echo "Starting AI Resort FastAPI backend..."

uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
