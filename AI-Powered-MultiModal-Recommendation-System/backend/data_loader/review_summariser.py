"""
backend/review_summariser.py
------------------------------
Generates CAUTIOUS, hallucination-resistant review summaries.

The core problem with embedding raw reviews:
  1. Reviews are mixed — same restaurant gets 1★ and 5★
  2. Reviews can be fake (competitor attacks, paid reviews)
  3. Outdated reviews reflect old management / old kitchen
  4. Embedding noisy text produces noisy vectors

Our approach:
  - Feed all reviews for a restaurant to an LLM
  - Prompt it to write a CAUTIOUS 2-sentence summary
  - The prompt explicitly instructs: only state what MULTIPLE
    reviews agree on, flag disagreements, never fabricate
  - The summary is then embedded (not the raw reviews)
  - The summary is shown to users WITH a disclaimer

This means our review signal is:
  "Several reviewers mentioned good biryani, though opinions
   on service are mixed. Some recent reviews mention long wait times."

NOT:
  "This restaurant has amazing biryani and great service."
"""

import os
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

_groq = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY", ""),
    model="llama-3.3-70b-versatile",
    temperature=0.1,    # low temperature — we want factual, not creative
)

_gemini = ChatGoogleGenerativeAI(
    api_key=os.environ.get("GOOGLE_API_KEY", ""),
    model="gemini-1.5-flash",
    temperature=0.1,
)


def _invoke_llm(messages: list) -> str:
    try:
        return _groq.invoke(messages).content
    except Exception as e:
        print(f"[review_summariser] Groq failed ({e}), switching to Gemini...")
        return _gemini.invoke(messages).content


SYSTEM_PROMPT = """You are a cautious review analyst. Your job is to summarise
customer reviews for a restaurant in exactly 2 sentences.

STRICT RULES — violating these makes the output useless:
1. Only state what MULTIPLE reviews consistently agree on.
2. If reviews are mixed or contradictory on a point, say so explicitly.
3. Never fabricate details not present in the reviews.
4. Never use superlatives (amazing, best, perfect, worst).
5. Flag if review count is very low (fewer than 3) — low confidence.
6. Flag if ratings are highly polarised (many 1★ and 5★) — suspicious pattern.
7. Write in third person. No "I" or "we".

Output format: exactly 2 sentences. Nothing else. No labels."""


def summarise_reviews(
    restaurant_name: str,
    cuisine: str,
    reviews: list[dict],
    max_reviews: int = 20
) -> dict:
    """
    Generate a cautious review summary for embedding and display.

    Args:
        restaurant_name: Name of the restaurant
        cuisine:         Cuisine type
        reviews:         List of review dicts with keys:
                         reviewer_name, rating, text, published_date, source
        max_reviews:     Cap to avoid token overflow (default 20)

    Returns:
        {
            "summary":        str — the 2-sentence cautious summary
            "confidence":     "low" | "medium" | "high"
            "review_count":   int
            "avg_rating":     float | None
            "polarised":      bool — True if ratings are suspiciously spread
            "disclaimer":     str — shown to users alongside the summary
        }
    """
    if not reviews:
        return {
            "summary":      "No reviews available for this restaurant.",
            "confidence":   "none",
            "review_count": 0,
            "avg_rating":   None,
            "polarised":    False,
            "disclaimer":   "No review data available."
        }

    # Cap reviews to avoid token overflow
    reviews = reviews[:max_reviews]
    review_count = len(reviews)

    # Compute stats
    ratings = [r.get("rating") for r in reviews if r.get("rating") is not None]
    avg_rating = round(sum(ratings) / len(ratings), 1) if ratings else None

    # Detect polarisation: many 1s and 5s with few in between
    polarised = False
    if len(ratings) >= 5:
        extremes = sum(1 for r in ratings if r <= 2 or r >= 4.5)
        polarised = (extremes / len(ratings)) > 0.7

    # Confidence level
    if review_count < 3:
        confidence = "low"
    elif review_count < 8:
        confidence = "medium"
    else:
        confidence = "high"

    # Build compact review text — strip null fields, cap text length
    review_lines = []
    for r in reviews:
        rating = r.get("rating", "?")
        text   = (r.get("text") or "").strip()[:200]   # hard cap per review
        date   = r.get("published_date", "")[:10] if r.get("published_date") else ""
        if text:
            line = f"[{rating}★{' ' + date if date else ''}] {text}"
            review_lines.append(line)

    if not review_lines:
        return {
            "summary":      "Reviews exist but contain no text content.",
            "confidence":   "low",
            "review_count": review_count,
            "avg_rating":   avg_rating,
            "polarised":    polarised,
            "disclaimer":   "Reviews had no text content to analyse."
        }

    reviews_text = "\n".join(review_lines)

    user_prompt = f"""Restaurant: {restaurant_name} ({cuisine})
Number of reviews: {review_count}
Average rating: {avg_rating or 'unknown'}
Polarised ratings: {'Yes — treat with extra caution' if polarised else 'No'}

Reviews:
{reviews_text}

Write a cautious 2-sentence summary following the rules."""

    summary = _invoke_llm([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])

    # Build disclaimer shown to users
    disclaimer_parts = [f"Based on {review_count} review{'s' if review_count != 1 else ''}."]
    if confidence == "low":
        disclaimer_parts.append("Low confidence — very few reviews.")
    if polarised:
        disclaimer_parts.append("⚠️ Reviews are highly polarised — may include fake reviews.")

    return {
        "summary":      summary.strip(),
        "confidence":   confidence,
        "review_count": review_count,
        "avg_rating":   avg_rating,
        "polarised":    polarised,
        "disclaimer":   " ".join(disclaimer_parts)
    }
