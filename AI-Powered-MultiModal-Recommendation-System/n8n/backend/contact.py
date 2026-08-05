"""
backend/contact.py
-------------------
Two responsibilities:

1. generate_contact_message()
   Uses Groq/Gemini to write a natural, polite contact message
   on behalf of the user to a restaurant.

2. trigger_n8n_contact()
   Calls the n8n webhook with full restaurant + message data.
   n8n then routes to email / WhatsApp / booking based on
   what contact info the restaurant has.
"""

import os
import httpx
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from backend.db.database import AsyncSessionLocal
from backend.model.models import Restaurant, Review
from backend.retrieving import vector_store, bm25_store
from backend.data_loader.apify_loader import normalize_apify_place   # reuse existing logic

load_dotenv()


APIFY_ACTOR_ID = "compass~crawler-google-places"   # Google Maps scraper
APIFY_RUN_SYNC_URL = f"https://api.apify.com/v2/acts/{APIFY_ACTOR_ID}/run-sync-get-dataset-items"

DEFAULT_CITIES = ["Lahore", "Islamabad", "Karachi", "Rawalpindi"]
DEFAULT_PER_CITY_LIMIT = 50


# ── LLM setup ──────────────────────────────────────────────────────────────

_groq = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY", ""),
    model="llama-3.3-70b-versatile",
    temperature=0.7,
)

_gemini = ChatGoogleGenerativeAI(
    api_key=os.environ.get("GOOGLE_API_KEY", ""),
    model="gemini-1.5-flash",
    temperature=0.7,
)


def _invoke_llm(messages: list) -> str:
    try:
        return _groq.invoke(messages).content
    except Exception as e:
        print(f"[contact] Groq failed ({e}), switching to Gemini...")
        return _gemini.invoke(messages).content


# ── Message generator ──────────────────────────────────────────────────────

def generate_contact_message(
    restaurant_name: str,
    cuisine: str,
    city: str,
    user_name: str,
    user_query: str,
    contact_method: str          # "email" | "whatsapp" | "booking"
) -> str:
    """
    Generate a polite, natural contact message from the user to the restaurant.

    The message is tailored to:
      - The restaurant's name and cuisine
      - What the user was searching for (their original query)
      - The contact channel (email is formal, WhatsApp is casual)

    Args:
        restaurant_name: Name of the restaurant being contacted
        cuisine:         Cuisine type of the restaurant
        city:            City the restaurant is in
        user_name:       Name of the user sending the message
        user_query:      What the user originally searched for
        contact_method:  "email" | "whatsapp" | "booking"

    Returns:
        A ready-to-send message string the user can approve or edit.
    """
    tone = {
        "email":    "formal and professional",
        "whatsapp": "friendly and conversational, keep it brief",
        "booking":  "formal and concise, focus on reservation details"
    }.get(contact_method, "polite and friendly")

    system_prompt = """You are writing a contact message from a customer to a restaurant.
Write ONLY the message body — no subject line, no labels, no explanation.
The message should sound natural, like a real person wrote it.
Keep it under 100 words."""

    user_prompt = f"""Write a {tone} message from {user_name} to {restaurant_name},
a {cuisine} restaurant in {city}.

The customer was looking for: "{user_query}"

The message should:
- Greet the restaurant
- Mention what they are interested in (based on their search)
- Ask if the restaurant can accommodate them
- End politely with the customer's name

Write ONLY the message. Nothing else."""

    return _invoke_llm([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])


# ── n8n webhook caller ─────────────────────────────────────────────────────

def trigger_n8n_contact(
    restaurant_id: int,
    restaurant_name: str,
    cuisine: str,
    city: str,
    email: str | None,
    phone: str | None,
    website: str | None,
    message: str,
    user_name: str,
    user_query: str
) -> dict:
    """
    Send restaurant contact data to the n8n webhook.

    n8n receives this payload and routes to:
      - Email    if restaurant has email
      - WhatsApp if restaurant has phone (via WhatsApp Business API)
      - Booking  if restaurant has website with booking

    The routing logic lives entirely in n8n — we just send the full
    payload and let n8n decide.

    Args:
        All restaurant contact fields + the approved message + user info.

    Returns:
        Dict with status and n8n response.
    """
    n8n_webhook_url = os.environ.get("N8N_WEBHOOK_URL", "")

    if not n8n_webhook_url:
        return {
            "success": False,
            "error": "N8N_WEBHOOK_URL not set in environment variables."
        }

    # Determine available contact methods for n8n routing
    contact_methods = []
    if email:
        contact_methods.append("email")
    if phone:
        contact_methods.append("whatsapp")
    if website:
        contact_methods.append("booking")

    payload = {
        "restaurant": {
            "id":      restaurant_id,
            "name":    restaurant_name,
            "cuisine": cuisine,
            "city":    city,
            "email":   email,
            "phone":   phone,
            "website": website,
        },
        "contact_methods": contact_methods,   # n8n uses this for routing
        "message":         message,
        "user": {
            "name":  user_name,
            "query": user_query,
        },
    }

    try:
        response = httpx.post(
            n8n_webhook_url,
            json=payload,
            timeout=10.0,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()

        return {
            "success":         True,
            "contact_methods": contact_methods,
            "n8n_status":      response.status_code,
            "message":         f"Contact request sent via {', '.join(contact_methods) or 'no channel available'}."
        }

    except httpx.HTTPStatusError as e:
        return {
            "success": False,
            "error":   f"n8n webhook returned {e.response.status_code}: {e.response.text}"
        }
    except Exception as e:
        return {
            "success": False,
            "error":   str(e)
        }



def _get_apify_token() -> str:
    token = os.environ.get("APIFY_API_TOKEN", "").strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "APIFY_API_TOKEN not set. Get one free at apify.com "
                "(Settings → Integrations → API token) and add it to .env"
            )
        )
    return token


# ── Apify API call ──────────────────────────────────────────────────────────

async def _run_apify_scraper(cities: list[str], per_city_limit: int) -> list[dict]:
    """
    Trigger the Apify Google Maps scraper actor synchronously and
    return the raw scraped places.

    Uses run-sync-get-dataset-items — this blocks until the actor
    finishes and returns results directly, no polling required.
    Can take 1-5 minutes depending on how many cities/results requested.
    """
    token = _get_apify_token()

    search_terms = [f"restaurants in {city} Pakistan" for city in cities]

    actor_input = {
        "searchStringsArray": search_terms,
        "maxCrawledPlacesPerSearch": per_city_limit,
        "language": "en",
        "exportPlaceUrls": False,
        "skipClosedPlaces": True,
    }

    async with httpx.AsyncClient(timeout=600.0) as client:   # 10 min timeout — actor runs can be slow
        try:
            response = await client.post(
                APIFY_RUN_SYNC_URL,
                params={"token": token},
                json=actor_input,
            )
            response.raise_for_status()
            return response.json()   # list of raw place dicts

        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Apify API error {e.response.status_code}: {e.response.text[:300]}"
            )
        except httpx.TimeoutException:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=(
                    "Apify actor run timed out (10 min limit). "
                    "Try reducing per_city_limit or number of cities."
                )
            )

