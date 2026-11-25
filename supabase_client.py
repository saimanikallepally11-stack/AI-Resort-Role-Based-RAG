
import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

# Load local .env file when running on your computer.
# Render will use its configured environment variables.
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Read Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Validate configuration
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError(
        "SUPABASE_URL or SUPABASE_KEY is missing. "
        "Configure these environment variables."
    )

# Create Supabase connection
supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

def verify_user_token(token):
    """
    Verify the Supabase access token.
    Returns the authenticated user if valid.
    """

    if not token:
        raise ValueError("Access token is missing")

    response = supabase.auth.get_user(token)

    if not response or not response.user:
        raise ValueError("Invalid authentication token")

    return response.user
