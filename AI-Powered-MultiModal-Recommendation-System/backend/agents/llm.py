"""
backend/core/config.py
-----------------------
Configuration and agent execution helper supporting Groq and Gemini APIs.
"""

import os
from groq import Groq
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

# Initialise clients using environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Map specific agents to preferred provider/model, or use defaults
AGENT_MODEL_MAP = {
    "profile_analyser": {"provider": "gemini", "model": "gemini-3.6-flash"},
    "candidate_retriever": {"provider": "groq", "model": "openai/gpt-oss-120b"},
    "trend_analyst": {"provider": "gemini", "model": "gemini-3.6-flash"},
    "style_expert": {"provider": "gemini", "model": "gemini-3.6-flash"},
    "nutrition_expert": {"provider": "groq", "model": "openai/gpt-oss-120b"},
    "reranker": {"provider": "groq", "model": "openai/gpt-oss-120b"},
}

def call_agent(agent_name: str, message: str) -> str:
    """
    Routes agent prompt to either Groq or Gemini based on configuration mapping.
    Falls back gracefully if a provider is unconfigured.
    """
    config = AGENT_MODEL_MAP.get(agent_name, {"provider": "groq", "model": "openai/gpt-oss-120b"})
    provider = config["provider"]
    model = config["model"]

    # Call Groq API
    if provider == "groq" and groq_client:
        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert restaurant recommendation agent. Follow instructions precisely and return valid formats when requested."},
                    {"role": "user", "content": message}
                ],
                temperature=0.3,
            )
            msg = response.choices[0].message
    
            # 2. Extract final answer, fall back to reasoning if content is empty
            final_text = getattr(msg, 'content', '') or getattr(msg, 'reasoning_content', '')
            
            return final_text.strip() if final_text else "Error: Empty response from model."
        except Exception as e:
            print(f"[call_agent] Groq error for {agent_name}: {e}")

    # Call Gemini API
    if provider == "gemini" and gemini_client:
        try:
            response = gemini_client.models.generate_content(
                model=model,
                contents=message,
                config=types.GenerateContentConfig(
                    system_instruction="You are an expert restaurant recommendation agent. Follow instructions precisely and return valid formats when requested.",
                    temperature=0.3,
                )
            )
            return response.text.strip()
        except Exception as e:
            print(f"[call_agent] Gemini error for {agent_name}: {e}")

    # Fallback cross-provider if primary fails or is missing
    if groq_client:
        try:
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": message}],
                temperature=0.3,
            )
            msg = response.choices[0].message
    
            # 2. Extract final answer, fall back to reasoning if content is empty
            final_text = getattr(msg, 'content', '') or getattr(msg, 'reasoning_content', '')
            
            return final_text.strip() if final_text else "Error: Empty response from model."
        except Exception as e:
            print(f"[call_agent] Groq fallback error: {e}")

    if gemini_client:
        try:
            response = gemini_client.models.generate_content(
                model="gemini-3.6-flash",
                contents=message
            )
            return response.text.strip()
        except Exception as e:
            print(f"[call_agent] Gemini fallback error: {e}")

    raise RuntimeError(f"Failed to execute agent '{agent_name}'. Check API keys for Groq and Gemini.")