"""
backend/data_enrichment.py
---------------------------
Enriches sparse OSM data before embedding.

OSM gives us: name, amenity tag (mapped to generic cuisine), coordinates.
We use three enrichment steps:

1. CUISINE NORMALISATION
   Maps OSM tags like "Coffee_Shop", "Bubble_Tea", "Regional" to
   proper food-type labels. "Restaurant" gets replaced by inference.

2. NAME-BASED CUISINE INFERENCE
   When cuisine is still "Restaurant" after normalisation,
   we scan the name for known keywords:
     "biryani" → Biryani
     "karahi"  → Pakistani
     "chinese" → Chinese
     "pizza"   → Pizza
     etc.

3. RICH EMBEDDING TEXT CONSTRUCTION
   Instead of:
     "Nando's is a Chicken restaurant in Islamabad."
   We produce:
     "Nando's is a grilled chicken and peri-peri restaurant in Islamabad,
      located in F-7 Markaz area. Known for: spicy chicken, burgers.
      Contact available. Open: check website."

This gives the embedding model real semantic content to work with
so similar queries match relevant restaurants even without descriptions.
"""

import re
import json

# ── OSM cuisine tag normalisation map ─────────────────────────────────────
# OSM uses technical tags — map them to user-friendly food types

OSM_CUISINE_MAP = {
    # Generic (needs inference)
    "restaurant":    None,    # trigger name inference
    "food_court":    "Food Court",

    # Beverages
    "cafe":          "Cafe",
    "coffee_shop":   "Coffee Shop",
    "bubble_tea":    "Bubble Tea",
    "juice_bar":     "Juice Bar",
    "juice":         "Juice Bar",

    # Fast food
    "fast_food":     "Fast Food",
    "burger":        "Burgers",
    "pizza":         "Pizza",
    "sandwich":      "Sandwiches",
    "donut":         "Donuts",
    "ice_cream":     "Ice Cream",

    # Protein-specific
    "chicken":       "Fried Chicken",
    "seafood":       "Seafood",
    "kebab":         "Kebabs",
    "bbq":           "BBQ & Grill",

    # Cuisines
    "chinese":       "Chinese",
    "japanese":      "Japanese",
    "thai":          "Thai",
    "indian":        "Indian",
    "pakistani":     "Pakistani",
    "arabic":        "Arabic",
    "middle_eastern":"Middle Eastern",
    "mediterranean": "Mediterranean",
    "italian":       "Italian",
    "american":      "American",
    "mexican":       "Mexican",
    "turkish":       "Turkish",
    "korean":        "Korean",

    # Local
    "regional":      "Regional Pakistani",
    "biryani":       "Biryani",
    "karahi":        "Karahi & Pakistani",
    "halwa_puri":    "Halwa Puri",
    "nihari":        "Nihari",

    # Bakery
    "bakery":        "Bakery",
    "pastry":        "Pastry & Bakery",

    # Misc
    "asian":         "Asian",
    "european":      "European",
    "international": "International",
    "steakhouse":    "Steakhouse",
    "sushi":         "Japanese",
}


# ── Name-based cuisine inference keywords ──────────────────────────────────
# Checked in order — first match wins

NAME_CUISINE_PATTERNS = [
    # Pakistani / South Asian
    (r'\bbiryani\b',          "Biryani"),
    (r'\bkarahi\b',           "Pakistani Karahi"),
    (r'\bnihari\b',           "Pakistani Nihari"),
    (r'\bshanwari\b',         "Peshwari / Chapli Kebab"),
    (r'\bpulao\b',            "Pakistani Pulao"),
    (r'\bkebab|kabab\b',      "Kebabs & BBQ"),
    (r'\bsavour\b',           "Pakistani Fast Food"),
    (r'\btandoor\b',          "Tandoori"),
    (r'\bdesi\b',             "Desi / Pakistani"),
    (r'\balladdin|aladdin\b', "Pakistani"),
    (r'\bmandi\b',            "Arabic Mandi"),

    # Fast food chains (global)
    (r'\bmcdonald|mcdonalds\b',    "Burgers"),
    (r'\bkfc|kentucky\b',          "Fried Chicken"),
    (r'\bnando\b',                 "Peri-Peri Chicken"),
    (r'\bsubway\b',                "Sandwiches"),
    (r'\bdominos|domino\b',        "Pizza"),
    (r'\bpizza hut\b',             "Pizza"),
    (r'\b14th street pizza\b',     "Pizza"),

    # Food types
    (r'\bpizza\b',            "Pizza"),
    (r'\bburger\b',           "Burgers"),
    (r'\bshawarma|shwarma\b', "Shawarma"),
    (r'\bchicken\b',          "Fried Chicken"),
    (r'\bsushi\b',            "Japanese"),
    (r'\bramen|noodle\b',     "Asian Noodles"),
    (r'\bchinese\b',          "Chinese"),
    (r'\bjapanese\b',         "Japanese"),
    (r'\bkorean\b',           "Korean"),
    (r'\bthai\b',             "Thai"),
    (r'\bitalian\b',          "Italian"),
    (r'\bpasta\b',            "Italian"),

    # Beverages
    (r'\bcoffee|cafe|kafe\b', "Cafe & Coffee"),
    (r'\btea|chai\b',         "Tea & Chai"),
    (r'\bgelato|ice.?cream\b', "Ice Cream & Gelato"),
    (r'\bjuice\b',            "Juice Bar"),
    (r'\bbubble.?tea\b',      "Bubble Tea"),
    (r'\bdonut|doughnut\b',   "Donuts"),

    # Bakery
    (r'\bbakery|bakers\b',    "Bakery"),
    (r'\bcookie|cookies\b',   "Bakery"),

    # Grill / BBQ
    (r'\bbbq|barbeque|grill\b', "BBQ & Grill"),
    (r'\bsteakhouse|steak\b',   "Steakhouse"),

    # Dragon = usually Chinese
    (r'\bdragon\b',           "Chinese"),
    (r'\bwok\b',              "Asian Stir Fry"),

    # Seafood
    (r'\bseafood|fish\b',     "Seafood"),

    # Wraps / sandwiches
    (r'\bwrap\b',             "Wraps & Sandwiches"),
]


def normalise_cuisine(raw_cuisine: str, name: str = "") -> str:
    """
    Convert a raw OSM cuisine tag to a clean, useful label.

    Priority:
      1. Direct OSM tag mapping
      2. Name-based keyword inference (when tag is generic)
      3. "Restaurant" as final fallback (still better than None)

    Args:
        raw_cuisine: The raw cuisine string from OSM / DB
        name:        Restaurant name (used for inference)

    Returns:
        Clean cuisine string like "Biryani", "Chinese", "Cafe & Coffee"
    """
    raw = (raw_cuisine or "").strip().lower()

    # Step 1: direct map
    mapped = OSM_CUISINE_MAP.get(raw)
    if mapped:
        return mapped

    # Step 2: tag is "restaurant" or unmapped → infer from name
    if not mapped or raw in ("restaurant", ""):
        inferred = _infer_from_name(name)
        if inferred:
            return inferred

    # Step 3: title-case the raw tag if it's something specific
    if raw and raw not in ("restaurant", ""):
        return raw.replace("_", " ").title()

    return "Restaurant"


def _infer_from_name(name: str) -> str | None:
    """Scan restaurant name for cuisine keywords. Returns None if no match."""
    name_lower = (name or "").lower()
    for pattern, cuisine in NAME_CUISINE_PATTERNS:
        if re.search(pattern, name_lower):
            return cuisine
    return None


# ── Address / area extraction ──────────────────────────────────────────────

# Islamabad sector / area patterns
ISLAMABAD_AREAS = [
    r'\b[FGHI]-\d+\b',           # F-6, G-9, H-8 etc.
    r'\bblue area\b',
    r'\bdha\b',
    r'\bmargalla\b',
    r'\bbahria\b',
    r'\be-\d+\b', r'\bd-\d+\b',
    r'\bsatellite town\b',
    r'\brawal\w+\b',
    r'\bchaklala\b',
]

LAHORE_AREAS = [
    r'\bgulberg\b', r'\bdha\b', r'\bjohur town\b',
    r'\bmodel town\b', r'\bcantt\b', r'\bwapda town\b',
    r'\bliberty\b', r'\bmm alam\b',
]

KARACHI_AREAS = [
    r'\bclifton\b', r'\bdefence\b', r'\bdha\b',
    r'\bkorangi\b', r'\bgulshan\b', r'\bmalir\b',
    r'\bsaddar\b', r'\bnazimabad\b',
]


def extract_area(address: str, city: str) -> str | None:
    """Extract a known area/sector from an address string."""
    if not address:
        return None

    addr_lower = address.lower()
    patterns = {
        "Islamabad":  ISLAMABAD_AREAS,
        "Rawalpindi": ISLAMABAD_AREAS,   # shares some areas
        "Lahore":     LAHORE_AREAS,
        "Karachi":    KARACHI_AREAS,
    }.get(city, [])

    for pattern in patterns:
        match = re.search(pattern, addr_lower, re.IGNORECASE)
        if match:
            return match.group(0).strip().upper()

    return None


# ── Rich embedding text builder ────────────────────────────────────────────

# What each cuisine type is known for — adds semantic richness
# when there's no description in the data
CUISINE_CONTEXT = {
    "Biryani":            "rice, spices, meat, aromatic Pakistani biryani",
    "Pakistani Karahi":   "karahi, spicy, Pakistani desi food, wok cooking",
    "Burgers":            "burgers, fries, fast food, beef chicken patty",
    "Pizza":              "pizza, italian, cheese, dough, toppings",
    "Cafe & Coffee":      "coffee, espresso, latte, cappuccino, light meals, pastries",
    "Coffee Shop":        "coffee, espresso, latte, cappuccino, cozy seating",
    "Bubble Tea":         "bubble tea, boba, tapioca pearls, milk tea",
    "Chinese":            "chinese food, noodles, dim sum, stir fry, wok",
    "Japanese":           "japanese food, sushi, ramen, sashimi, miso",
    "Shawarma":           "shawarma, wrap, garlic sauce, chicken beef",
    "Fried Chicken":      "fried chicken, crispy, spicy, fast food",
    "Peri-Peri Chicken":  "peri peri, grilled chicken, spicy, portuguese",
    "BBQ & Grill":        "BBQ, grilled meat, seekh kebab, tikka, tandoor",
    "Fast Food":          "fast food, quick meals, affordable, takeaway",
    "Donuts":             "donuts, doughnuts, glazed, pastry, sweet",
    "Ice Cream & Gelato": "ice cream, gelato, frozen dessert, cold sweet",
    "Bakery":             "bakery, bread, pastry, cakes, baked goods",
    "Sandwiches":         "sandwiches, wraps, subs, deli, fresh bread",
    "Juice Bar":          "fresh juice, smoothies, healthy drinks, fruits",
    "Seafood":            "seafood, fish, prawns, grilled, fresh catch",
    "Arabic Mandi":       "mandi, arabic food, slow cooked lamb rice",
    "Asian":              "asian food, rice noodles mixed cuisine",
    "Korean":             "korean food, bulgogi, bibimbap, kimchi, k-food",
    "Thai":               "thai food, pad thai, curry, spicy, aromatic",
    "Italian":            "italian food, pasta, pizza, risotto, tiramisu",
}


def build_rich_embedding_text(
    name: str,
    cuisine: str,
    city: str,
    address: str = None,
    description: str = None,
    tags: str = None,
    opening_hours: str = None,
    phone: str = None,
    website: str = None,
    rating: float = None,
) -> str:
    """
    Build a semantically rich text string for embedding.

    Instead of: "Nando's is a Chicken restaurant in Islamabad."
    Produces:   "Nando's is a peri-peri grilled chicken restaurant in Islamabad,
                 Agha Khan Road area. Known for: peri peri, grilled chicken,
                 spicy, portuguese. Contact available."

    This gives the sentence-transformer model real content to encode,
    making similar queries match relevant restaurants.

    Args:
        All available restaurant fields from PostgreSQL.

    Returns:
        A single text string ready to be embedded.
    """
    # Normalise cuisine
    clean_cuisine = normalise_cuisine(cuisine, name)

    # Extract area from address
    area = extract_area(address or "", city)

    # Build location string
    location_parts = [city]
    if area:
        location_parts.append(f"{area} area")
    elif address:
        # Use first meaningful part of address (not just numbers)
        addr_clean = re.sub(r'^\d+[,\s]*', '', address.strip())
        if len(addr_clean) > 3:
            location_parts.append(addr_clean[:50])
    location_str = ", ".join(location_parts)

    # Start building text
    parts = [f"{name} is a {clean_cuisine} restaurant in {location_str}."]

    # Add description if available (Apify data will have this)
    if description and len(description.strip()) > 10:
        parts.append(description.strip()[:250])

    # Add cuisine context (fills the gap when description is missing)
    elif clean_cuisine in CUISINE_CONTEXT:
        parts.append(f"Known for: {CUISINE_CONTEXT[clean_cuisine]}.")

    # Add tags
    if tags:
        try:
            tag_list = json.loads(tags)
            if tag_list:
                parts.append(f"Features: {', '.join(str(t) for t in tag_list[:8])}.")
        except Exception:
            pass

    # Add availability signals (helps queries like "open late" or "has website")
    availability = []
    if phone:
        availability.append("phone contact available")
    if website:
        availability.append("has website")
    if opening_hours:
        try:
            hours_data = json.loads(opening_hours) if isinstance(opening_hours, str) else opening_hours
            raw = hours_data.get("raw", "") if isinstance(hours_data, dict) else ""
            if "24/7" in str(raw):
                availability.append("open 24 hours")
            elif raw:
                availability.append(f"hours: {raw[:40]}")
        except Exception:
            pass

    if availability:
        parts.append(". ".join(availability).capitalize() + ".")

    # Rating signal
    if rating and rating > 0:
        parts.append(f"Rated {rating}/5.")

    return " ".join(parts)
