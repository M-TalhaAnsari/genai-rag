"""
backend/agents/reranker.py
---------------------------
Standalone reranker — calls only Agent 6 on a pre-built candidate list.

Use this when you already have hybrid search results and just want
agent-generated reasoning and reranking without running the full workflow.

The full workflow (all 6 agents) lives in workflow.py.
This module is a lighter alternative for cases where:
  - The profile is unknown (anonymous user)
  - Speed matters more than depth of analysis
  - You want to add reasoning to already-retrieved results
"""

import json
from agents.llm import call_agent


def rerank_with_reasoning(
    query: str,
    candidates: list[dict],
    profile_summary: str = ""
) -> list[dict]:
    """
    Take a pre-retrieved candidate list and produce a final ranked output
    with written reasoning for each result.

    Args:
        query:          The original user search query.
        candidates:     List of restaurant dicts from hybrid_search() or
                        the full workflow's filtered_candidates.
        profile_summary: Plain-English profile summary from Agent 1,
                         or empty string for anonymous users.

    Returns:
        List of ranked recommendation dicts, each with a "reasoning" field.
        Falls back to the input order with generic reasoning on parse failure.
    """
    if not candidates:
        return []

    profile_context = (
        f"User profile summary:\n{profile_summary}\n\n"
        if profile_summary
        else "No user profile available — base recommendations on query relevance only.\n\n"
    )

    message = f"""You are producing final restaurant recommendations.

User query: "{query}"

{profile_context}Candidates to rank (already filtered for relevance):
{json.dumps(candidates, indent=2)}

Produce a ranked list of the top 5 (or fewer if fewer candidates exist).

Rules:
- Rank from best (#1) to weakest
- Each reasoning: 2-3 specific sentences — never generic filler
- Reference actual data: query terms, cuisine type, city, score
- Return ONLY a valid JSON array, no text outside it

Format:
[
  {{
    "rank": 1,
    "restaurant_id": 42,
    "name": "Sushi House",
    "cuisine": "Japanese",
    "city": "Rawalpindi",
    "rrf_score": 0.032,
    "reasoning": "Your reasoning here."
  }}
]
"""
    response = call_agent("reranker", message)

    try:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        return json.loads(cleaned.strip())

    except Exception as e:
        print(f"[reranker] Parse error: {e} — returning candidates with fallback reasoning")
        return [
            {
                "rank": i + 1,
                "restaurant_id": c.get("restaurant_id"),
                "name": c.get("name"),
                "cuisine": c.get("cuisine"),
                "city": c.get("city"),
                "rrf_score": c.get("rrf_score"),
                "reasoning": f"Matched '{query}' via hybrid search (rank {i + 1})."
            }
            for i, c in enumerate(candidates[:5])
        ]
