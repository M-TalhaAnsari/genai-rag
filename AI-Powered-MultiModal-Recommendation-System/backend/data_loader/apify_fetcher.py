
import os
import httpx
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from urllib.parse import quote, urlencode
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.db_models import Restaurant
import json
from backend.models.db_models import ApifyRawSnapshot
from backend.models.db_models import ApifyRunLog
from backend.data_loader.apify_loader import normalize_apify_place

from dotenv import load_dotenv
load_dotenv()

import asyncio 

from fastapi import HTTPException, status

from backend.core.database import AsyncSessionLocal

from sqlalchemy import select



APIFY_ACTOR_ID = "compass~crawler-google-places"

APIFY_START_RUN_URL     = f"https://api.apify.com/v2/acts/{APIFY_ACTOR_ID}/runs"
APIFY_RUN_STATUS_URL    = "https://api.apify.com/v2/actor-runs/{run_id}"
APIFY_DATASET_ITEMS_URL = "https://api.apify.com/v2/datasets/{dataset_id}/items"


TARGET_COUNTRY_CODE = "pk"

DATA_DIR = "data/apify_raw"

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
    model="gemini-3.5-flash",
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

async def _run_apify_targeted_by_place_ids(
    place_ids: list[str],
    max_reviews: int = 5,
    max_images: int = 3,
) -> tuple[list[dict], dict, int, str]:
    """
    Scrapes specific known places directly by Google placeId — no search
    phase, no risk of rediscovering restaurants you already have as 'new'.
    Only pulls the detail pass (reviews, photos, hours) for places you
    already identified as incomplete.
    """
    token = _get_apify_token()

    actor_input = {
        "placeIds":                  place_ids,   # direct lookup, not search
        "maxReviews":                max_reviews,
        "maxImages":                 max_images,
        "reviewsSort":               "newest",
        "scrapeReviewsPersonalData": False,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        start_resp = await client.post(
            APIFY_START_RUN_URL, params={"token": token}, json=actor_input
        )
        start_resp.raise_for_status()
        run_data   = start_resp.json().get("data", {})
        run_id     = run_data.get("id")
        dataset_id = run_data.get("defaultDatasetId")

        final_status = "UNKNOWN"
        for attempt in range(MAX_POLL_ATTEMPTS):
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            status_resp = await client.get(
                APIFY_RUN_STATUS_URL.format(run_id=run_id), params={"token": token}
            )
            status_data = status_resp.json().get("data", {})
            final_status = status_data.get("status", "UNKNOWN")
            if final_status in TERMINAL_STATUSES:
                break

        is_partial = final_status != "SUCCEEDED"
        items_resp = await client.get(
            APIFY_DATASET_ITEMS_URL.format(dataset_id=dataset_id),
            params={"token": token, "clean": "true"}, timeout=60.0
        )
        raw_places = items_resp.json()

    snapshot_id = await _save_raw_snapshot(raw_places, run_id) if raw_places else 0
    run_info = {"status": final_status, "is_partial": is_partial, "message": ""}
    return raw_places, run_info, snapshot_id, run_id


async def _merge_enrichment_into_existing(raw_places: list[dict]) -> dict:

    updated, reviews_added, errors = 0, 0, []

    async with AsyncSessionLocal() as db:
        for place in raw_places:
            try:
                restaurant_data, reviews = normalize_apify_place(place)
                external_id = restaurant_data.get("external_id")
                if not external_id:
                    continue

                result = await db.execute(
                    select(Restaurant).where(Restaurant.external_id == external_id)
                )
                existing = result.scalars().first()
                if not existing:
                    continue   # not one of ours — ignore, don't insert new rows here

                if restaurant_data.get("photos"):
                    existing.photos = restaurant_data["photos"]
                if restaurant_data.get("opening_hours"):
                    existing.opening_hours = restaurant_data["opening_hours"]
                if restaurant_data.get("review_count"):
                    existing.review_count = restaurant_data["review_count"]
                if restaurant_data.get("rating"):
                    existing.rating = restaurant_data["rating"]

                for rv in reviews:
                    existing_rv = await db.execute(
                        select(Review).where(
                            Review.restaurant_id == existing.id,
                            Review.text == rv.get("text"),
                        )
                    )
                    if existing_rv.scalars().first():
                        continue
                    db.add(Review(restaurant_id=existing.id, **rv))
                    reviews_added += 1

                updated += 1
            except Exception as e:
                errors.append(f"{place.get('title', '?')}: {e}")

        await db.commit()

    return {"updated": updated, "reviews_added": reviews_added, "errors": errors[:10]}

# ── Apify API call ──────────────────────────────────────────────────────────
TARGET_COUNTRY_CODE = "pk"   # was lowercase "pk" — keep consistent with post-filter check

DATA_DIR = "data/apify_raw"

async def _save_raw_snapshot(raw_places: list[dict], run_id: str) -> int:
    async with AsyncSessionLocal() as db:
        snapshot = ApifyRawSnapshot(
            run_id=run_id,
            raw_json=json.dumps(raw_places, ensure_ascii=False),
            place_count=len(raw_places),
            processed=False,
        )
        db.add(snapshot)
        await db.commit()
        await db.refresh(snapshot)
        return snapshot.id


async def _mark_snapshot_processed(snapshot_id: int) -> None:
    from sqlalchemy import update
    from backend.models.db_models import ApifyRawSnapshot

    async with AsyncSessionLocal() as db:
        await db.execute(
            update(ApifyRawSnapshot)
            .where(ApifyRawSnapshot.id == snapshot_id)
            .values(processed=True)
        )
        await db.commit()


async def _log_run(
    run_id: str,
    search_terms: list[str],
    per_city_limit: int,
    places_fetched: int,
    new_inserted: int,
    duplicates: int,
    is_partial: bool,
) -> None:

    async with AsyncSessionLocal() as db:
        db.add(ApifyRunLog(
            run_id=run_id,
            search_terms=json.dumps(search_terms),
            per_city_limit=per_city_limit,
            places_fetched=places_fetched,
            new_inserted=new_inserted,
            duplicates=duplicates,
            is_partial=is_partial,
        ))
        await db.commit()


async def _run_apify_scraper_resilient(
    search_terms: list[str],
    per_city_limit: int,
    max_reviews: int = 5,
    max_images: int = 3,
) -> tuple[list[dict], dict, int, str]:
    """
    Runs the Apify actor with the given search terms, polls to completion
    or timeout, fetches the dataset regardless of final status, and saves
    a raw snapshot to Neon before returning.

    Returns:
        (raw_places, run_info, snapshot_id, run_id)
    """
    token = _get_apify_token()

    actor_input = {
        "searchStringsArray":        search_terms,
        "maxCrawledPlacesPerSearch": per_city_limit,
        "language":                  "en",
        "countryCode":               TARGET_COUNTRY_CODE,
        "maxReviews":                max_reviews,
        "maxImages":                 max_images,
        "reviewsSort":               "newest",
        "scrapeReviewsPersonalData": False,
        "exportPlaceUrls":           False,
        "skipClosedPlaces":          True,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
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
                detail=f"Apify failed to start run: {e.response.status_code} {e.response.text[:3000]}"
            )

        run_data   = start_resp.json().get("data", {})
        run_id     = run_data.get("id")
        dataset_id = run_data.get("defaultDatasetId")

        if not run_id or not dataset_id:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Apify run started but returned no run ID / dataset ID."
            )

        print(f"[apify] Run started: {run_id} (dataset: {dataset_id})")

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

    snapshot_id = await _save_raw_snapshot(raw_places, run_id) if raw_places else 0

    run_info = {
        "status":     final_status,
        "is_partial": is_partial,
        "message":    status_message or (
            "Run did not complete normally — data may be incomplete, "
            "but everything scraped before it stopped was recovered."
            if is_partial else "Run completed successfully."
        )
    }

    print(f"[apify] Recovered {len(raw_places)} places (partial={is_partial}), snapshot_id={snapshot_id}")
    return raw_places, run_info, snapshot_id, run_id