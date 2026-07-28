"""
mcp_service/mcp_server.py
---------------------------
FastMCP server exposing three tools:
  - get_restaurant_info   — lookup by name
  - recommend_by_vibe     — search by atmosphere keyword
  - get_review            — retrieve a user review

Also exposes the raw culinary map as an MCP resource.

Run standalone:
    python mcp_service/mcp_server.py
"""

import json
from pathlib import Path

from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

mcp = FastMCP("Connoisseur-Server")

DATA_DIR = Path(__file__).parent.parent / "data"
CULINARY_MAP_PATH = DATA_DIR / "California-Culinary-Map.txt"
RESTAURANT_DATA_PATH = DATA_DIR / "structured_restaurant_data.json"
REVIEW_DATA_PATH = DATA_DIR / "augmented_user_review.json"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_restaurants() -> list[dict]:
    with open(RESTAURANT_DATA_PATH, "r") as f:
        return json.load(f)


def _load_reviews() -> list[dict]:
    with open(REVIEW_DATA_PATH, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# MCP Resource
# ---------------------------------------------------------------------------

@mcp.resource("culinary-map://california")
def get_culinary_map() -> str:
    """The full raw California Culinary Map text.
    Contains detailed descriptions of 100+ restaurants across California
    including their vibes, cuisines, ratings, and price ranges."""
    return CULINARY_MAP_PATH.read_text()


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def get_restaurant_info(restaurant_name: str) -> str:
    """Search for a restaurant by name and return its structured information."""
    restaurants = _load_restaurants()
    query = restaurant_name.lower().strip()

    matches = [
        r for r in restaurants
        if query in r.get("name", "").lower() or r.get("name", "").lower() in query
    ]

    if not matches:
        return json.dumps({
            "status": "not_found",
            "message": (
                f"No restaurant found matching '{restaurant_name}'. "
                "Try a partial name like 'Iron' or 'Sakura'."
            ),
        }, indent=2)

    return json.dumps(
        {"status": "found", "count": len(matches), "results": matches},
        indent=2,
    )


@mcp.tool()
def recommend_by_vibe(vibe: str) -> str:
    """Find restaurants matching a given vibe or atmosphere keyword.

    Example keywords: moody, sun-drenched, romantic, family-friendly,
    trendy, cozy, rustic, modern, casual, upscale.
    """
    restaurants = _load_restaurants()
    vibe_lower = vibe.lower().strip()

    structured_matches = []
    for r in restaurants:
        vibes_list = [v.lower() for v in r.get("vibes", [])]
        description = r.get("description", "").lower()
        if any(vibe_lower in v for v in vibes_list) or vibe_lower in description:
            structured_matches.append({
                "name": r.get("name"),
                "vibes": r.get("vibes"),
                "description": r.get("description"),
                "cuisine": r.get("cuisine"),
                "rating": r.get("rating"),
                "price_range": r.get("price_range"),
            })

    raw_text = CULINARY_MAP_PATH.read_text().lower()
    text_excerpts = [
        para.strip()[:300]
        for para in raw_text.split("\n\n")
        if vibe_lower in para and para.strip()
    ]

    return json.dumps({
        "vibe_searched": vibe,
        "structured_matches": structured_matches,
        "raw_text_excerpts": text_excerpts,
    }, indent=2)


@mcp.tool()
def get_review(restaurant_name: str) -> str:
    """Retrieve the full user review for a restaurant."""
    reviews = _load_reviews()
    query = restaurant_name.lower().strip()

    matching_review = None
    for review in reviews:
        if query in review.get("restaurant_name", "").lower():
            matching_review = review
            break

    if not matching_review:
        return json.dumps({
            "status": "not_found",
            "message": f"No review found for '{restaurant_name}'.",
        }, indent=2)

    return json.dumps({
        "status": "found",
        "restaurant": matching_review["restaurant_name"],
        "reviewer": matching_review.get("reviewer"),
        "rating": matching_review.get("rating"),
        "review_text": matching_review.get("review_text"),
        "image_description": matching_review.get("image_description", "N/A"),
        "visit_date": matching_review.get("visit_date", "N/A"),
    }, indent=2)
