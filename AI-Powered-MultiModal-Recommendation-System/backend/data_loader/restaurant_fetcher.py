"""
backend/restaurant_fetcher.py
------------------------------
Weekly incremental sync — fetches new restaurants from two free sources:

  1. OpenStreetMap Overpass API  — no key, completely free
  2. Foursquare Places API       — free tier: 100k calls/month

These run on n8n schedule (weekly) to catch newly opened restaurants.
Apify handles the one-time bulk load of rich data (see apify_loader.py).

Fields extracted per source:
  OSM:        name, cuisine, city, address, phone, website, hours, lat/lon
  Foursquare: name, cuisine, city, address, phone, website, rating, hours, lat/lon

Cities: Lahore, Islamabad, Karachi, Rawalpindi
"""

import json
import os
import time
from typing import Optional

import httpx

# ── City config ────────────────────────────────────────────────────────────

CITY_BBOXES = {
    "Lahore":     (31.3780, 74.1543, 31.6340, 74.4580),
    "Islamabad":  (33.5700, 72.8200, 33.7650, 73.2100),
    "Karachi":    (24.7900, 66.9700, 25.1700, 67.4600),
    "Rawalpindi": (33.5100, 72.9900, 33.7200, 73.1500),
}

CITY_CENTRES = {
    "Lahore":     (31.5204, 74.3587),
    "Islamabad":  (33.6844, 73.0479),
    "Karachi":    (24.8607, 67.0011),
    "Rawalpindi": (33.5651, 73.0169),
}

TARGET_CITIES = list(CITY_BBOXES.keys())

OVERPASS_URL    = "https://overpass-api.de/api/interpreter"
FOURSQUARE_URL  = "https://api.foursquare.com/v3/places/search"


# ── OpenStreetMap Overpass ─────────────────────────────────────────────────

def _overpass_query(bbox: tuple) -> str:
    s, w, n, e = bbox
    return f"""
[out:json][timeout:30];
(
  node["amenity"="restaurant"]({s},{w},{n},{e});
  node["amenity"="cafe"]({s},{w},{n},{e});
  node["amenity"="fast_food"]({s},{w},{n},{e});
  node["amenity"="food_court"]({s},{w},{n},{e});
);
out body;
"""


def _parse_osm_hours(tags: dict) -> Optional[str]:
    raw = tags.get("opening_hours")
    if not raw:
        return None
    return json.dumps({"raw": raw})


def _normalize_osm(element: dict, city: str) -> Optional[dict]:
    tags = element.get("tags", {})
    name = tags.get("name", "").strip()
    if not name:
        return None

    raw_cuisine = tags.get("cuisine", "")
    cuisine = raw_cuisine.split(";")[0].strip().title() if raw_cuisine else (
        {"cafe": "Cafe", "fast_food": "Fast Food",
         "food_court": "Food Court"}.get(tags.get("amenity", ""), "Restaurant")
    )

    return {
        "name":          name,
        "cuisine":       cuisine,
        "city":          city,
        "address":       tags.get("addr:full") or tags.get("addr:street"),
        "phone":         tags.get("phone") or tags.get("contact:phone"),
        "website":       tags.get("website") or tags.get("contact:website"),
        "menu_url":      tags.get("menu:url") or tags.get("contact:menu"),
        "latitude":      element.get("lat"),
        "longitude":     element.get("lon"),
        "opening_hours": _parse_osm_hours(tags),
        "source":        "openstreetmap",
        "external_id":   str(element.get("id")),
    }


def fetch_city_osm(city: str) -> list[dict]:
    bbox = CITY_BBOXES.get(city)
    if not bbox:
        return []

    try:
        response = httpx.post(
            OVERPASS_URL,
            data={"data": _overpass_query(bbox)},
            timeout=35.0,
            headers={"User-Agent": "ConnoisseurApp/1.0"}
        )
        response.raise_for_status()
        elements = response.json().get("elements", [])
    except Exception as e:
        print(f"[osm] Error for {city}: {e}")
        return []

    seen, results = set(), []
    for el in elements:
        record = _normalize_osm(el, city)
        if record and record["name"] not in seen:
            seen.add(record["name"])
            results.append(record)

    time.sleep(1)   # Overpass fair-use rate limit
    print(f"[osm] {city}: {len(results)} restaurants")
    return results


# ── Foursquare Places API ──────────────────────────────────────────────────

def _parse_fsq_hours(place: dict) -> Optional[str]:
    hours = place.get("hours")
    if not hours:
        return None
    return json.dumps(hours)


def _normalize_foursquare(place: dict, city: str) -> Optional[dict]:
    name = place.get("name", "").strip()
    if not name:
        return None

    categories = place.get("categories", [])
    cuisine = categories[0].get("name", "Restaurant") if categories else "Restaurant"

    loc = place.get("location", {})
    address = loc.get("formatted_address") or loc.get("address")

    geo = place.get("geocodes", {}).get("main", {})

    price_map = {1: "$", 2: "$$", 3: "$$$", 4: "$$$$"}
    price = price_map.get(place.get("price"))

    return {
        "name":          name,
        "cuisine":       cuisine,
        "city":          loc.get("locality") or city,
        "address":       address,
        "phone":         place.get("tel"),
        "website":       place.get("website"),
        "menu_url":      None,
        "latitude":      geo.get("latitude"),
        "longitude":     geo.get("longitude"),
        "rating":        place.get("rating"),
        "review_count":  place.get("stats", {}).get("total_ratings"),
        "price_level":   price,
        "description":   place.get("description"),
        "opening_hours": _parse_fsq_hours(place),
        "photos":        None,   # requires separate Foursquare photos endpoint
        "source":        "foursquare",
        "external_id":   place.get("fsq_id"),
    }


def fetch_city_foursquare(city: str, limit: int = 50) -> list[dict]:
    api_key = os.environ.get("FOURSQUARE_API_KEY", "").strip()
    if not api_key:
        return []

    centre = CITY_CENTRES.get(city)
    if not centre:
        return []

    lat, lon = centre

    try:
        response = httpx.get(
            FOURSQUARE_URL,
            params={
                "query":      "restaurant",
                "ll":         f"{lat},{lon}",
                "radius":     15000,
                "categories": "13065",
                "limit":      min(limit, 50),
                "sort":       "RATING",
                "fields":     "name,categories,location,geocodes,tel,website,"
                              "rating,stats,price,description,hours,fsq_id",
            },
            headers={
                "Authorization": api_key,
                "Accept":        "application/json",
            },
            timeout=10.0
        )
        response.raise_for_status()
        places = response.json().get("results", [])
    except Exception as e:
        print(f"[foursquare] Error for {city}: {e}")
        return []

    results = []
    for place in places:
        record = _normalize_foursquare(place, city)
        if record:
            results.append(record)

    print(f"[foursquare] {city}: {len(results)} restaurants")
    return results


# ── Combined fetcher ───────────────────────────────────────────────────────

def fetch_city(city: str) -> list[dict]:
    """Fetch from OSM + Foursquare, deduplicate by name."""
    osm = fetch_city_osm(city)
    fsq = fetch_city_foursquare(city)

    seen = {r["name"].lower() for r in osm}
    combined = list(osm)
    for r in fsq:
        if r["name"].lower() not in seen:
            seen.add(r["name"].lower())
            combined.append(r)

    print(f"[fetcher] {city}: {len(combined)} total")
    return combined


def fetch_all_cities(cities: list[str] = None) -> list[dict]:
    cities = cities or TARGET_CITIES
    all_restaurants = []
    for city in cities:
        all_restaurants.extend(fetch_city(city))
    print(f"[fetcher] Grand total: {len(all_restaurants)} restaurants")
    return all_restaurants
