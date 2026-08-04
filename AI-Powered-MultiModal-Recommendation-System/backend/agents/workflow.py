"""
backend/agents/workflow.py
---------------------------
Multi-agent recommendation workflow.

Phase 1 — sequential:
    Agent 1: profile_analyser   → summarises user taste from DB profile
    Agent 2: candidate_retriever → calls hybrid_search, filters bad results

Phase 2 — parallel (ThreadPoolExecutor):
    Agent 3: trend_analyst      → flags trendy candidates
    Agent 4: style_expert       → matches candidates to user flavor profile
    Agent 5: nutrition_expert   → checks dietary fit

Phase 3 — sequential:
    Agent 6: reranker           → synthesises all insights → final ranked list

State dict is passed through every phase.
Each node reads from state and writes back to it.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from backend.agents.configs import call_agent
from backend import retrieval as retrieval_module


# ── State initialiser ──────────────────────────────────────────────────────

def _initial_state(query: str, user_id: str, profile: dict | None) -> dict:
    return {
        "query": query,
        "user_id": user_id,
        "user_profile": profile or {},

        # Phase 1 outputs
        "profile_summary": "",
        "filtered_candidates": [],

        # Phase 2 outputs
        "trend_analysis": "",
        "style_analysis": "",
        "nutrition_analysis": "",

        # Phase 3 output
        "final_recommendations": [],

        "workflow_step": "start",
    }


# ── Phase 1 nodes ──────────────────────────────────────────────────────────

def node_analyse_profile(state: dict) -> dict:
    """
    Agent 1 — Profile Analyser.
    Reads the user profile dict and produces a plain-English summary
    that later agents can use as context.
    """
    print("[workflow] Phase 1a: analysing user profile...")

    profile = state["user_profile"]

    if not profile:
        state["profile_summary"] = "No profile data available. Treat this as a new user with no known preferences."
        state["workflow_step"] = "profile_analysed"
        return state

    message = f"""Analyse this user preference profile and write a concise summary
(maximum 150 words) of their dining personality for use by other recommendation agents.

User profile:
{json.dumps(profile, indent=2)}

Include:
- Their top cuisine preferences (be specific)
- Cities they prefer dining in
- Any cuisines they actively avoid
- Overall adventurousness (conservative / moderate / adventurous)
- Any other strong signals from the data

Write in third person: "This user prefers..."
"""
    state["profile_summary"] = call_agent("profile_analyser", message)
    state["workflow_step"] = "profile_analysed"
    print("[workflow] Profile summary done.")
    return state


def node_retrieve_and_filter(state: dict) -> dict:
    """
    Agent 2 — Candidate Retriever.
    Runs hybrid_search to get raw candidates, then asks the agent
    to filter out clearly irrelevant results.
    """
    print("[workflow] Phase 1b: retrieving and filtering candidates...")

    # Call the actual hybrid retrieval (BM25 + ChromaDB + RRF)
    raw_candidates = retrieval_module.hybrid_search(
        query=state["query"],
        top_k=20   # fetch more than needed — agent will filter down
    )

    if not raw_candidates:
        state["filtered_candidates"] = []
        state["workflow_step"] = "candidates_retrieved"
        print("[workflow] No candidates found.")
        return state

    message = f"""You are reviewing search results for the following user query:
"{state['query']}"

User profile summary:
{state['profile_summary']}

Raw candidates from hybrid search (ranked by RRF score):
{json.dumps(raw_candidates, indent=2)}

Your task:
1. Remove any candidates that clearly do not match the query or user profile.
2. Keep at least 5 results, maximum 10.
3. Return ONLY a valid JSON array of the kept candidates.
4. Each item must preserve all original fields plus add a "filter_note" field
   explaining in one sentence why it was kept.
5. Return ONLY the JSON array. No explanation outside the JSON.

Example format:
[
  {{
    "restaurant_id": 1,
    "name": "Sushi House",
    "cuisine": "Japanese",
    "city": "Rawalpindi",
    "rrf_score": 0.032,
    "dense_rank": 1,
    "sparse_rank": 2,
    "filter_note": "Directly matches the user's query for Japanese food."
  }}
]
"""
    response = call_agent("candidate_retriever", message)

    try:
        # Strip markdown code fences if the LLM wraps the JSON
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        state["filtered_candidates"] = json.loads(cleaned.strip())
    except Exception as e:
        print(f"[workflow] Filter parse error: {e} — using raw candidates")
        state["filtered_candidates"] = raw_candidates[:10]

    state["workflow_step"] = "candidates_retrieved"
    print(f"[workflow] {len(state['filtered_candidates'])} candidates after filtering.")
    return state


# ── Phase 2 nodes (run in parallel) ───────────────────────────────────────

def node_analyse_trends(state: dict) -> dict:
    """Agent 3 — Trend Analyst."""
    print("[workflow] Phase 2a: trend analysis...")

    message = f"""User query: "{state['query']}"

Candidate restaurants:
{json.dumps(state['filtered_candidates'], indent=2)}

Analyse which of these candidates align with current food trends in Pakistan
(Lahore, Islamabad, Karachi, Rawalpindi).

Write 3-5 sentences identifying:
- Which candidates represent trendy or emerging dining concepts
- Which align with popular cuisine movements in these cities
- Any standout options worth highlighting for trend reasons

Be specific about restaurant names.
"""
    state["trend_analysis"] = call_agent("trend_analyst", message)
    print("[workflow] Trend analysis done.")
    return state


def node_analyse_styles(state: dict) -> dict:
    """Agent 4 — Style Expert."""
    print("[workflow] Phase 2b: style analysis...")

    message = f"""User query: "{state['query']}"

User profile summary:
{state['profile_summary']}

Candidate restaurants:
{json.dumps(state['filtered_candidates'], indent=2)}

Analyse the food styles and flavour profiles of these candidates.
Match each to the user's known preferences from their profile.

Write 3-5 sentences identifying:
- Which candidates best match the user's demonstrated taste
- Which offer the flavour profiles implied by the query
- Any candidates that are a surprisingly good stylistic fit

Be specific about restaurant names and cuisine styles.
"""
    state["style_analysis"] = call_agent("style_expert", message)
    print("[workflow] Style analysis done.")
    return state


def node_evaluate_nutrition(state: dict) -> dict:
    """Agent 5 — Nutrition Expert."""
    print("[workflow] Phase 2c: nutrition evaluation...")

    message = f"""User query: "{state['query']}"

User profile summary:
{state['profile_summary']}

Candidate restaurants:
{json.dumps(state['filtered_candidates'], indent=2)}

Check these candidates for dietary suitability based on the user profile.

Write 2-4 sentences covering:
- Any candidates that conflict with known dietary restrictions or strong dislikes
- Which candidates are clearly safe and appropriate
- Any cuisine types that warrant a caution flag

If no dietary information is available in the profile, state that clearly
and give a general assessment based on cuisine type.
"""
    state["nutrition_analysis"] = call_agent("nutrition_expert", message)
    print("[workflow] Nutrition evaluation done.")
    return state


# ── Phase 3 node ───────────────────────────────────────────────────────────

def node_rerank_and_explain(state: dict) -> dict:
    """
    Agent 6 — Reranker and Explainer.
    Synthesises all Phase 2 insights into a final ranked list
    with written reasoning for each result.
    """
    print("[workflow] Phase 3: reranking and generating explanations...")

    message = f"""You are producing the final restaurant recommendations for this user.

User query: "{state['query']}"

User profile summary:
{state['profile_summary']}

Filtered candidates:
{json.dumps(state['filtered_candidates'], indent=2)}

Trend analysis:
{state['trend_analysis']}

Style analysis:
{state['style_analysis']}

Nutrition/dietary assessment:
{state['nutrition_analysis']}

Your task:
Produce a final ranked list of the top 5 recommendations.

Rules:
- Rank from best match (#1) to weakest match (#5)
- Each reasoning must be 2-3 specific sentences — never generic
- Cite actual signals: query match, profile preferences, trend fit, style fit
- Do not recommend anything flagged as a dietary conflict
- Return ONLY a valid JSON array. No text outside the JSON.

Required format:
[
  {{
    "rank": 1,
    "restaurant_id": 42,
    "name": "Sushi House",
    "cuisine": "Japanese",
    "city": "Rawalpindi",
    "rrf_score": 0.032,
    "reasoning": "Directly matches your search for Japanese food in Rawalpindi.
You have liked Japanese restaurants before, and this is the top-ranked result
from both semantic and keyword search — a strong signal of relevance."
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
        state["final_recommendations"] = json.loads(cleaned.strip())
    except Exception as e:
        print(f"[workflow] Reranker parse error: {e} — returning filtered candidates")
        # Graceful fallback: return filtered candidates without agent reasoning
        state["final_recommendations"] = [
            {
                "rank": i + 1,
                "restaurant_id": c.get("restaurant_id"),
                "name": c.get("name"),
                "cuisine": c.get("cuisine"),
                "city": c.get("city"),
                "rrf_score": c.get("rrf_score"),
                "reasoning": "Matched your query via hybrid search."
            }
            for i, c in enumerate(state["filtered_candidates"][:5])
        ]

    state["workflow_step"] = "complete"
    print(f"[workflow] Final recommendations: {len(state['final_recommendations'])} results.")
    return state


# ── Main entry point ───────────────────────────────────────────────────────

def run_recommendation_workflow(
    query: str,
    user_id: str,
    profile: dict | None = None
) -> dict:
    """
    Execute the full multi-agent recommendation pipeline.

    Args:
        query:   The user's natural language search query.
        user_id: Used for logging and profile lookup.
        profile: Pre-loaded user profile dict from feedback.get_profile().
                 Pass None for anonymous / new users.

    Returns:
        State dict containing final_recommendations and all intermediate outputs.
    """
    state = _initial_state(query, user_id, profile)

    # ── Phase 1: sequential ──────────────────────────────────────────────
    state = node_analyse_profile(state)
    state = node_retrieve_and_filter(state)

    if not state["filtered_candidates"]:
        state["final_recommendations"] = []
        state["workflow_step"] = "complete"
        return state

    # ── Phase 2: parallel ────────────────────────────────────────────────
    print("[workflow] Phase 2: running trend, style, nutrition in parallel...")

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(node_analyse_trends,    dict(state)): "trend_analysis",
            executor.submit(node_analyse_styles,    dict(state)): "style_analysis",
            executor.submit(node_evaluate_nutrition, dict(state)): "nutrition_analysis",
        }
        for future in as_completed(futures):
            key = futures[future]
            result_state = future.result()
            state[key] = result_state[key]

    # ── Phase 3: sequential ──────────────────────────────────────────────
    state = node_rerank_and_explain(state)

    return state
