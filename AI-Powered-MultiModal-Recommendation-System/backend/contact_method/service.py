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
from urllib.parse import quote, urlencode
from sqlalchemy.ext.asyncio import AsyncSession
from backend.model.models import Restaurant

import asyncio 

from fastapi import HTTPException, status

from backend.db.database import AsyncSessionLocal


load_dotenv()


APIFY_ACTOR_ID = "compass~crawler-google-places"

APIFY_START_RUN_URL     = f"https://api.apify.com/v2/acts/{APIFY_ACTOR_ID}/runs"
APIFY_RUN_STATUS_URL    = "https://api.apify.com/v2/actor-runs/{run_id}"
APIFY_DATASET_ITEMS_URL = "https://api.apify.com/v2/datasets/{dataset_id}/items"


TARGET_COUNTRY_CODE = "pk"

# Polling config
POLL_INTERVAL_SECONDS = 10
MAX_POLL_ATTEMPTS = 60   # 60 × 10s = 10 min max wait

TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"}

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

from langchain_core.messages import SystemMessage, HumanMessage
import json


def generate_contact_messages(
    restaurant_name: str,
    cuisine: str,
    city: str,
    user_name: str,
    user_query: str
) -> dict:
    """
    Generate contact messages for both email and WhatsApp.

    Args:
        restaurant_name: Name of restaurant
        cuisine: Restaurant cuisine type
        city: Restaurant city
        user_name: Customer name
        user_query: Original user request/search

    Returns:
        Dictionary containing email and whatsapp messages.
    """

    system_prompt = """
You are an assistant that writes customer messages to restaurants.

Generate two versions:

1. Email:
- Formal and professional
- Include greeting
- Explain customer requirement clearly
- Ask if restaurant can accommodate
- End politely with customer's name
- Maximum 120 words

2. WhatsApp:
- Friendly and conversational
- Short and natural
- Suitable for WhatsApp chat
- Ask the restaurant about availability
- End politely with customer's name
- Maximum 60 words


Return ONLY valid JSON.
No markdown.
No explanation.

Format:

{
    "email_message": "...",
    "whatsapp_message": "..."
}
"""


    user_prompt = f"""
Create messages for this customer:

Customer name:
{user_name}

Restaurant:
{restaurant_name}

Cuisine:
{cuisine}

City:
{city}

Customer request:
{user_query}

Generate both email and WhatsApp messages.
"""


    response = _invoke_llm([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])


    try:
        messages = json.loads(response)

    except json.JSONDecodeError:
        raise ValueError(
            "LLM returned invalid JSON response"
        )


    return {
        "email_message": messages["email_message"],
        "whatsapp_message": messages["whatsapp_message"]
    }




async def generate_whatsapp_url(
    db: AsyncSession,
    restaurant: Restaurant,
    message: str,
) -> dict:

    if not restaurant.phone:
        return {
            "success": False,
            "error": "Restaurant has no WhatsApp/phone number."
        }

    phone = "".join(filter(str.isdigit, restaurant.phone))

    if phone.startswith("0"):
        phone = "92" + phone[1:]

    whatsapp_url = (
        f"https://wa.me/{phone}"
        f"?text={quote(message)}"
    )

    return {
        "success": True,
        "channel": "whatsapp",
        "action": "manual",
        "restaurant_id": restaurant.id,
        "restaurant_name": restaurant.name,
        "whatsapp_url": whatsapp_url,
    }


def generate_gmail_compose_url(
    recipient_email: str,
    message: str,
    subject: str = "Reservation Request"
) -> str:
    """
    Generate Gmail compose URL.

    User clicks this URL:
    - Gmail opens
    - Recipient is filled
    - Subject is filled
    - Message is filled
    """

    params = {
        "view": "cm",
        "fs": "1",
        "to": recipient_email,
        "su": subject,
        "body": message
    }

    return (
        "https://mail.google.com/mail/?"
        + urlencode(params)
    )

# ── n8n webhook caller ─────────────────────────────────────────────────────

# def trigger_n8n_contact(
#     restaurant_id: int,
#     restaurant_name: str,
#     cuisine: str,
#     city: str,
#     email: str | None,
#     phone: str | None,
#     website: str | None,
#     message: str,
#     user_name: str,
#     user_query: str
# ) -> dict:
#     """
#     Send restaurant contact data to the n8n webhook.

#     n8n receives this payload and routes to:
#       - Email    if restaurant has email
#       - WhatsApp if restaurant has phone (via WhatsApp Business API)
#       - Booking  if restaurant has website with booking

#     The routing logic lives entirely in n8n — we just send the full
#     payload and let n8n decide.

#     Args:
#         All restaurant contact fields + the approved message + user info.

#     Returns:
#         Dict with status and n8n response.
#     """
#     n8n_webhook_url = os.environ.get("N8N_WEBHOOK_URL", "")

#     if not n8n_webhook_url:
#         return {
#             "success": False,
#             "error": "N8N_WEBHOOK_URL not set in environment variables."
#         }

#     # Determine available contact methods for n8n routing
#     contact_methods = []
#     if email:
#         contact_methods.append("email")
#     if phone:
#         contact_methods.append("whatsapp")
#     if website:
#         contact_methods.append("booking")

#     payload = {
#         "restaurant": {
#             "id":      restaurant_id,
#             "name":    restaurant_name,
#             "cuisine": cuisine,
#             "city":    city,
#             "email":   email,
#             "phone":   phone,
#             "website": website,
#         },
#         "contact_methods": contact_methods,   # n8n uses this for routing
#         "message":         message,
#         "user": {
#             "name":  user_name,
#             "query": user_query,
#         },
#     }

#     try:
#         response = httpx.post(
#             n8n_webhook_url,
#             json=payload,
#             timeout=10.0,
#             headers={"Content-Type": "application/json"}
#         )
#         response.raise_for_status()

#         return {
#             "success":         True,
#             "contact_methods": contact_methods,
#             "n8n_status":      response.status_code,
#             "message":         f"Contact request sent via {', '.join(contact_methods) or 'no channel available'}."
#         }

#     except httpx.HTTPStatusError as e:
#         return {
#             "success": False,
#             "error":   f"n8n webhook returned {e.response.status_code}: {e.response.text}"
#         }
#     except Exception as e:
#         return {
#             "success": False,
#             "error":   str(e)
#         }



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


# ── Internal helper ────────────────────────────────────────────────────────

async def _get_restaurant_contact_method(restaurant_id: int) -> dict:
    """
    Look up a restaurant's available contact channels and return
    the best one for the draft message tone.
    """
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Restaurant).where(Restaurant.id == restaurant_id)
        )
        r = result.scalars().first()

    return {
        "email" : r.get("email",None),
        "phone" : r.get("phone",None),
        "website":r.get("website",None),
    }


# ── Apify API call ──────────────────────────────────────────────────────────

async def _run_apify_scraper_resilient(
    cities: list[str],
    per_city_limit: int
) -> tuple[list[dict], dict]:
    """
    Start an Apify run, poll it, and fetch dataset items no matter how
    the run ends. 

    Returns:
        (raw_places, run_info)
        raw_places: list of scraped place dicts (may be partial if the
                    run didn't finish — that's fine, we still use them)
        run_info:   {"status": str, "is_partial": bool, "message": str}
    """
    token = _get_apify_token()
    search_terms = [f"restaurants in {city} Pakistan" for city in cities]

    actor_input = {
        "searchStringsArray":        search_terms,
        "maxCrawledPlacesPerSearch": per_city_limit,
        "language":                  "en",
        "countryCode":               TARGET_COUNTRY_CODE,   # restrict to Pakistan
        "exportPlaceUrls":           False,
        "skipClosedPlaces":          True,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: start the run — returns immediately, does not wait
        try:
            start_resp = await client.post(
                APIFY_START_RUN_URL,
                params={"token": token},
                json=actor_input,
            )
            start_resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Apify failed to start run: {e.response.status_code} {e.response.text[:300]}"
            )

        run_data    = start_resp.json().get("data", {})
        run_id      = run_data.get("id")
        dataset_id  = run_data.get("defaultDatasetId")

        if not run_id or not dataset_id:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Apify run started but returned no run ID / dataset ID."
            )

        print(f"[apify] Run started: {run_id} (dataset: {dataset_id})")

        # Step 2: poll until terminal status or max attempts reached
        final_status = "UNKNOWN"
        status_message = ""

        for attempt in range(MAX_POLL_ATTEMPTS):
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

            try:
                status_resp = await client.get(
                    APIFY_RUN_STATUS_URL.format(run_id=run_id),
                    params={"token": token},
                )
                status_resp.raise_for_status()
                status_data = status_resp.json().get("data", {})
                final_status = status_data.get("status", "UNKNOWN")
                status_message = status_data.get("statusMessage", "") or ""

                print(f"[apify] Poll {attempt+1}/{MAX_POLL_ATTEMPTS}: {final_status}")

                if final_status in TERMINAL_STATUSES:
                    break

            except Exception as e:
                print(f"[apify] Poll error (continuing): {e}")
                continue

        is_partial = final_status != "SUCCEEDED"

        try:
            items_resp = await client.get(
                APIFY_DATASET_ITEMS_URL.format(dataset_id=dataset_id),
                params={"token": token, "clean": "true"},
                timeout=60.0
            )
            items_resp.raise_for_status()
            raw_places = items_resp.json()
        except Exception as e:
            raw_places = []
            status_message += f" | dataset fetch error: {e}"

    run_info = {
        "status":     final_status,
        "is_partial": is_partial,
        "message":    status_message or (
            "Run did not complete normally — data may be incomplete, "
            "but everything scraped before it stopped was recovered."
            if is_partial else "Run completed successfully."
        )
    }

    print(f"[apify] Recovered {len(raw_places)} places (partial={is_partial})")
    return raw_places, run_info

