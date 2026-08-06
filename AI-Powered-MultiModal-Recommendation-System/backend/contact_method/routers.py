"""
n8n/api/routers.py
------------------------------
Fully automated Apify data pipeline — NO manual export/import step.

This replaces the manual workflow (run actor on Apify website → download
JSON → save to data/apify_export.json → POST /load-apify) with a single
API call that n8n triggers on a schedule.

HOW IT WORKS
-------------
1. n8n calls POST /n8n/sync-apify (on whatever schedule you want —
   weekly, daily, etc.)
2. This endpoint calls the Apify API directly using APIFY_API_TOKEN
3. Apify runs the Google Maps scraper actor SYNCHRONOUSLY and returns
   the scraped restaurants directly in the response (no polling needed)
4. We normalise the data (same logic as apify_loader.py — reused, not
   duplicated) and insert into PostgreSQL + ChromaDB + BM25
5. Review summaries are generated automatically for new restaurants

SETUP REQUIRED
---------------
1. Sign up at apify.com (free, $5/month credits, no card)
2. Get your API token: Settings → Integrations → API token
3. Add to .env:  APIFY_API_TOKEN=your_token_here

n8n workflow (simple):
    Schedule Trigger (weekly)
        → POST http://your-server:8000/n8n/sync-apify
        → (optional) Slack/Email notification with the response summary

No manual file download. No manual upload. Fully automated.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import AsyncSessionLocal
from backend.model.models import Restaurant, Review
from backend.retrieving import vector_store, bm25_store
from backend.data_loader.apify_loader import  _load_places # reuse existing logic

from backend.model.schemas import(
    GenerateMessageRequest, ContactRequest
)
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.model.models import Restaurant
from contact_method.service import generate_whatsapp_url
from contact_method.service import generate_gmail_compose_url

from contact_method.schemas import (
    EmailComposeRequest,
    EmailComposeResponse,
    WhatsAppRequest,
)
from sqlalchemy import select


from contact_method.service import _run_apify_scraper_resilient

contact_router = APIRouter(prefix="/n8n", tags=["n8n automation"])

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


APIFY_ACTOR_ID = "compass~crawler-google-places"   # Google Maps scraper

DEFAULT_CITIES = ["Lahore", "Islamabad", "Karachi", "Rawalpindi","Murree","Abbotabad"]

APIFY_ACTOR_ID = "compass~crawler-google-places"

RECOMMENDED_INCREMENTAL_LIMIT = 15   # scheduled weekly re-sync — cheaper

@contact_router.post("/sync-apify")
async def sync_apify(
    cities: str = None,
    per_city_limit: int = RECOMMENDED_INCREMENTAL_LIMIT
):
    """
    Run through n8n

    Query params:
      cities:          comma-separated city names, default: all 4 cities
      per_city_limit:  max restaurants per city.
                        Recommended: 50 for your FIRST bootstrap run,
                        10-15 for regular scheduled re-syncs (cheaper —
                        Apify re-scrapes the same top-rated places every
                        time regardless of what you already have stored).

    Returns a summary including:
      run_status:               Apify's final run status
      is_partial:                True if credits ran out / run didn't finish
      raw_places_fetched:        how many places Apify actually returned
      inserted / skipped:        new vs already-in-database
      skipped_wrong_country:     places rejected for not being in Pakistan
      new_vs_duplicate_ratio:    helps you tune per_city_limit over time
    """
    city_list = (
        [c.strip() for c in cities.split(",") if c.strip()]
        if cities else DEFAULT_CITIES
    )

    print(f"[n8n-apify] Starting resilient sync for cities: {city_list}")

    raw_places, run_info = await _run_apify_scraper_resilient(city_list, per_city_limit)

    if not raw_places:
        return {
            "message":     "No places recovered from Apify.",
            "run_status":  run_info["status"],
            "run_message": run_info["message"],
            "cities":      city_list,
            "inserted":    0,
            "skipped":     0,
        }

    # Load into PostgreSQL + ChromaDB + BM25 (shared logic with manual loader)
    result = await _load_places(raw_places)

    total_processed = result["inserted"] + result["skipped"]
    duplicate_ratio = (
        round(result["skipped"] / total_processed * 100, 1)
        if total_processed > 0 else 0.0
    )

    return {
        "message":                 "Apify sync complete" if not run_info["is_partial"]
                                    else "Apify sync partially completed — credits or run may have been interrupted, but recovered data was saved.",
        "run_status":               run_info["status"],
        "is_partial":                run_info["is_partial"],
        "run_message":               run_info["message"],
        "cities":                    city_list,
        "raw_places_fetched":        len(raw_places),
        "new_vs_duplicate_ratio":    f"{duplicate_ratio}% were already in database",
        **result,
    }


# ── Status check ─────────────────────────────────────────────────────────

@contact_router.get("/apify-status")
async def apify_status():
    """Quick health check — current data volume without triggering a sync."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Restaurant))
        restaurants = result.scalars().all()

    total = len(restaurants)
    by_source = {}
    by_country_flag = {"pk_verified": 0, "unverified": 0}

    for r in restaurants:
        src = r.source or "unknown"
        by_source[src] = by_source.get(src, 0) + 1

    return {
        "total_restaurants": total,
        "by_source":         by_source,
        "chroma_vectors":    vector_store.collection_count(),
        "bm25_index_exists": bm25_store.index_exists(),
    }

@contact_router.post("/generate-message")
async def generate_message(request: GenerateMessageRequest):
    """
    Generate draft contact messages for email and WhatsApp.

    Called when the user clicks 'Select' on a restaurant card.
    Returns drafts that the user can approve or edit before sending.
    """

    from contact_method.service import generate_contact_messages
    from contact_method.service import _get_restaurant_contact_method


    # Check available restaurant contact options
    contact_info = await _get_restaurant_contact_method(
        request.restaurant_id
    )


    messages = generate_contact_messages(
        restaurant_name=request.restaurant_name,
        cuisine=request.cuisine,
        city=request.city,
        user_name=request.user_name,
        user_query=request.user_query
    )


    return {
        "restaurant_name": request.restaurant_name,

        "contact_method": contact_info.get("method"),

        "email_message": messages["email_message"],

        "whatsapp_message": messages["whatsapp_message"],

        "restaurant_website":contact_info.get("website",None),

    }


# @contact_router.post("/contact-restaurant")
# async def contact_restaurant(
#     request: ContactRequest,
#     db: AsyncSession = Depends(get_db)
# ):
#     """
#     Send the approved message to the restaurant via n8n.

#     n8n receives the full payload and routes to:
#       - Email    if restaurant.email is set
#       - WhatsApp if restaurant.phone is set
#       - Booking  if restaurant.website is set

#     The routing logic lives in n8n — this endpoint just fires the webhook.
#     """
#     from n8n.backend.contact import trigger_n8n_contact

#     result = trigger_n8n_contact(
#         restaurant_id=request.restaurant_id,
#         restaurant_name=request.restaurant_name,
#         cuisine=request.cuisine,
#         city=request.city,
#         email=request.email,
#         phone=request.phone,
#         website=request.website,
#         message=request.message,
#         user_name=request.user_name,
#         user_query=request.user_query
#     )

    

    # if not result["success"]:
    #     raise HTTPException(
    #         status_code=status.HTTP_502_BAD_GATEWAY,
    #         detail=result.get("error", "n8n webhook failed.")
    #     )

    # return result


@contact_router.post("/whatsapp-url/{restaurant_id}")
async def create_whatsapp_url(
    restaurant_id: int,
    request: WhatsAppRequest,
    db: AsyncSession = Depends(get_db),
):

    result = await db.execute(
        select(Restaurant).where(Restaurant.id == restaurant_id)
    )

    restaurant = result.scalar_one_or_none()

    if restaurant is None:
        raise HTTPException(404, "Restaurant not found.")

    response = await generate_whatsapp_url(
        db=db,
        restaurant=restaurant,
        message=request.message,
    )

    if not response["success"]:
        raise HTTPException(400, response["error"])

    return response


@contact_router.post(
    "/generate-compose-url",
    response_model=EmailComposeResponse
)
def create_email_url(
    data: EmailComposeRequest
):

    gmail_url = generate_gmail_compose_url(
        recipient_email=data.email,
        message=data.message,
        subject=data.subject
    )

    return {
        "email": data.email,
        "gmail_url": gmail_url
    }