"""
backend/models/schemas.py
--------------------------
All Pydantic request and response schemas.
One file — easy to find any schema without hunting across routers.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


# ── Shared ─────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


# ── Restaurant ─────────────────────────────────────────────────────────────

class RestaurantInput(BaseModel):
    name: str
    cuisine: str
    city: str
    area:          Optional[str]   = None
    postal_code:   Optional[str]   = None
    address:       Optional[str]   = None
    phone:         Optional[str]   = None
    email:         Optional[str]   = None
    website:       Optional[str]   = None
    menu_url:      Optional[str]   = None
    latitude:      Optional[float] = None
    longitude:     Optional[float] = None
    rating:        Optional[float] = None
    review_count:  Optional[int]   = None
    price_level:   Optional[str]   = None
    description:   Optional[str]   = None
    opening_hours: Optional[str]   = None
    photos:        Optional[str]   = None
    tags:          Optional[str]   = None
    all_cuisines:  Optional[str]   = None
    reviews_distribution: Optional[str] = None
    source:        Optional[str]   = None
    external_id:   Optional[str]   = None


class RestaurantRequest(BaseModel):
    restaurants: List[RestaurantInput]


class ReviewOut(BaseModel):
    id: int
    reviewer_name: Optional[str]
    rating:        Optional[float]
    text:          Optional[str]
    published_date: Optional[str]
    source:        Optional[str]


class RestaurantDetail(BaseModel):
    id: int
    name: str
    cuisine: str
    city: str
    area:          Optional[str]
    address:       Optional[str]
    phone:         Optional[str]
    email:         Optional[str]
    website:       Optional[str]
    menu_url:      Optional[str]
    latitude:      Optional[float]
    longitude:     Optional[float]
    rating:        Optional[float]
    review_count:  Optional[int]
    price_level:   Optional[str]
    description:   Optional[str]
    opening_hours: Optional[str]
    photos:        Optional[str]
    tags:          Optional[str]
    all_cuisines:  Optional[str]
    source:        Optional[str]
    is_embedded:   bool
    reviews:       List[ReviewOut] = []


class SyncResponse(BaseModel):
    message: str
    inserted_count: int
    skipped_count: int
    embedded_count: int
    inserted: List[str]
    skipped: List[str]


# ── Search ─────────────────────────────────────────────────────────────────

class SearchResult(BaseModel):
    restaurant_id: int
    name: str
    cuisine: str
    city: str
    rrf_score: float
    dense_rank:  Optional[int]
    sparse_rank: Optional[int]
    personalisation_boost: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    result_count: int
    personalised: bool
    results: List[SearchResult]


# ── Recommend ──────────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    query:   str
    user_id: Optional[str] = None
    top_k:   int           = Field(default=5, ge=1, le=10)


# ── Feedback ───────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    user_id:         str
    restaurant_id:   int
    restaurant_name: str
    cuisine:         str
    city:            str
    signal:          int           = Field(..., ge=-1, le=1)
    query:           Optional[str] = None


# ── Contact links (NEW) ────────────────────────────────────────────────────

class ContactLinksResponse(BaseModel):
    """
    Clickable contact options for a restaurant card.
    Only fields that actually have data are populated — the rest are None.
    Frontend shows a button only when the field is not None.
    """
    restaurant_id:   int
    restaurant_name: str

    # Email — opens mailto: with pre-filled subject + body
    email:           Optional[str] = None
    email_href:      Optional[str] = None   # "mailto:x@y.com?subject=...&body=..."

    # WhatsApp — opens wa.me link with pre-filled message
    phone:           Optional[str] = None
    whatsapp_href:   Optional[str] = None   # "https://wa.me/923001234567?text=..."

    # Website — just the URL, opens in new tab
    website:         Optional[str] = None

    # Menu — direct menu link if available
    menu_url:        Optional[str] = None

    # Summary of what's available
    available_channels: List[str] = []


class GenerateMessageRequest(BaseModel):
    restaurant_id:   int
    restaurant_name: str
    cuisine:         str
    city:            str
    user_name:       str
    user_query:      str
    contact_method:  str = "email"


class ContactRequest(BaseModel):
    restaurant_id:   int
    restaurant_name: str
    cuisine:         str
    city:            str
    email:           Optional[str] = None
    phone:           Optional[str] = None
    website:         Optional[str] = None
    message:         str
    user_name:       str
    user_query:      str