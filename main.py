
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import create_client

from supabase_client import (
    supabase,
    SUPABASE_URL,
    SUPABASE_KEY
)

from assistant import get_role_answer


# ---------------------------------------
# FASTAPI APPLICATION
# ---------------------------------------

app = FastAPI(
    title="AI Resort Role-Based RAG API",
    description="Secure Supabase authentication with LangChain and Gemini",
    version="2.4.0"
)


# ---------------------------------------
# CORS CONFIGURATION
# ---------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ai-resort-guest-experience-operations-platform-1.ai.studio"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


security = HTTPBearer()


# ---------------------------------------
# ROLE MAPPING
# ---------------------------------------

ROLE_MAPPING = {
    "admin": "operations",
    "guest": "guest",
    "front_desk": "front_desk"
}


# ---------------------------------------
# REQUEST MODEL
# ---------------------------------------

class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


# ---------------------------------------
# AUTHENTICATION
# ---------------------------------------

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials

    try:
        # 1. Verify Supabase access token
        auth_response = supabase.auth.get_user(token)

        if not auth_response or not auth_response.user:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired access token"
            )

        user = auth_response.user

        # 2. Create user-specific Supabase client
        user_client = create_client(
            SUPABASE_URL,
            SUPABASE_KEY
        )

        # 3. Forward JWT for Row Level Security
        user_client.postgrest.auth(token)

        # 4. Retrieve user's database role
        profile_response = (
            user_client
            .table("user")
            .select("role")
            .eq("auth_user_id", str(user.id))
            .limit(1)
            .execute()
        )

        profiles = profile_response.data or []
        profile = profiles[0] if profiles else None

        if not profile or not profile.get("role"):
            raise HTTPException(
                status_code=403,
                detail="User profile or role not found"
            )

        role = profile["role"].strip().lower()

        # 5. Verify role permissions
        if role not in ROLE_MAPPING:
            raise HTTPException(
                status_code=403,
                detail="Unauthorized user role"
            )

        return {
            "user_id": str(user.id),
            "email": user.email,
            "role": role
        }

    except HTTPException:
        raise

    except Exception as error:
        print("Authentication Error:", repr(error))

        raise HTTPException(
            status_code=401,
            detail="Authentication or profile verification failed"
        ) from error


# ---------------------------------------
# HOME ENDPOINT
# ---------------------------------------

@app.get("/")
def home():
    return {
        "message": "AI Resort Role-Based RAG API is running!",
        "status": "success",
        "supported_roles": list(ROLE_MAPPING.keys())
    }


# ---------------------------------------
# ROLE-BASED RAG ENDPOINT
# ---------------------------------------

@app.post("/ask")
def ask_question(
    request: QuestionRequest,
    current_user: dict = Depends(get_current_user)
):
    role_key = current_user["role"]

    # Convert database role into RAG access role
    rag_role = ROLE_MAPPING[role_key]

    try:
        result = get_role_answer(
            request.question,
            rag_role
        )

        return {
            "status": "success",
            "email": current_user["email"],
            "role": role_key,
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"]
        }

    except Exception as error:
        print("RAG API Error:", repr(error))

        raise HTTPException(
            status_code=500,
            detail="Unable to process your question. Please try again."
        ) from error
