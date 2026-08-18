"""
backend/core/config.py
-----------------------
Single source of truth for all environment variables.
Import from here everywhere — never use os.environ directly in routers/services.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Database
    NEON_DATABASE_URL: str = os.environ.get("NEON_DATABASE_URL", "")

    # LLMs
    GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
    GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")

    # Data sources
    APIFY_API_TOKEN: str = os.environ.get("APIFY_API_TOKEN", "")
    FOURSQUARE_API_KEY: str = os.environ.get("FOURSQUARE_API_KEY", "")

    # n8n
    N8N_WEBHOOK_URL: str = os.environ.get("N8N_WEBHOOK_URL", "")

    # App
    APP_VERSION: str = "4.0.0"
    APP_TITLE: str = "Connoisseur Restaurant API"
    APP_DESCRIPTION: str = "AI-powered restaurant discovery for Pakistan"

    REDIS_URL: str = "redis://localhost:6379/0"
    SESSION_TTL_SECONDS: int = 7200   
    GOOGLE_CLIENT_ID: str
    GOOGLE_CLIENT_SECRET: str
    GOOGLE_REDIRECT_URI: str = "http://127.0.0.1:8000/auth/google/callback"
    FRONTEND_URL: str = "http://localhost:8501"  

settings = Settings()