from pydantic import BaseModel, Field
from typing import Optional, List
# ── Request models ─────────────────────────────────────────────────────────

class RestaurantInput(BaseModel):
    name: str
    cuisine: str
    city: str
    address:       Optional[str]   = None
    phone:         Optional[str]   = None
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
    source:        Optional[str]   = None
    external_id:   Optional[str]   = None


class RestaurantRequest(BaseModel):
    restaurants: List[RestaurantInput]


class FeedbackRequest(BaseModel):
    user_id:         str
    restaurant_id:   int
    restaurant_name: str
    cuisine:         str
    city:            str
    signal:          int            = Field(..., ge=-1, le=1)
    query:           Optional[str]  = None


class RecommendRequest(BaseModel):
    query:   str
    user_id: Optional[str] = None
    top_k:   int           = Field(default=5, ge=1, le=10)


# ── Response models ────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


class SyncResponse(BaseModel):
    message: str
    inserted_count: int
    skipped_count: int
    embedded_count: int
    inserted: List[str]
    skipped: List[str]


class SearchResult(BaseModel):
    restaurant_id: int
    name: str
    cuisine: str
    city: str
    rrf_score: float
    dense_rank: Optional[int]
    sparse_rank: Optional[int]
    personalisation_boost: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    result_count: int
    personalised: bool
    results: List[SearchResult]


class ReviewOut(BaseModel):
    id: int
    reviewer_name: Optional[str]
    rating: Optional[float]
    text: Optional[str]
    published_date: Optional[str]
    source: Optional[str]


class RestaurantDetail(BaseModel):
    id: int
    name: str
    cuisine: str
    city: str
    address: Optional[str]
    phone: Optional[str]
    website: Optional[str]
    menu_url: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    rating: Optional[float]
    review_count: Optional[int]
    price_level: Optional[str]
    description: Optional[str]
    opening_hours: Optional[str]
    photos: Optional[str]
    tags: Optional[str]
    source: Optional[str]
    is_embedded: bool
    reviews: List[ReviewOut] = []


