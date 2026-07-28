"""
recommendation_engine/workflow.py
-----------------------------------
Multi-agent recommendation workflow.

Phases:
  1. User Analysis      — sequential
  2. Data Retrieval     — sequential
  3. Analysis           — parallel (trends, styles, nutrition)
  4. Synthesis          — sequential

LLM stack: Groq (primary) → Gemini (fallback).
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from recommendation_engine.agents import build_system_prompt

load_dotenv()

# ---------------------------------------------------------------------------
# LLM setup – Groq primary, Gemini fallback
# ---------------------------------------------------------------------------

groq_llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.7,
)

gemini_llm = ChatGoogleGenerativeAI(
    api_key=os.environ.get("GOOGLE_API_KEY"),
    model="gemini-1.5-flash",
    temperature=0.7,
)


def _invoke_llm(messages):
    """Call Groq; fall back to Gemini on any error."""
    try:
        return groq_llm.invoke(messages)
    except Exception as e:
        print(f"Groq failed ({e}), switching to Gemini…")
        return gemini_llm.invoke(messages)


# ---------------------------------------------------------------------------
# Agent caller
# ---------------------------------------------------------------------------

def call_agent(agent_key: str, user_message: str) -> str:
    """Invoke an agent by key, returning its raw text response."""
    system_prompt = build_system_prompt(agent_key)
    response = _invoke_llm([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ])
    return response.content


# ---------------------------------------------------------------------------
# Phase nodes
# ---------------------------------------------------------------------------

def node_generate_profile(state: dict) -> dict:
    print("\n[Phase 1] Generating user profile…")

    user_message = f"""Analyze this user data and create a comprehensive profile:

{state['user_input']}

Provide output in JSON format with these keys:
- favorite_cuisines (list)
- dietary_restrictions (list)
- dining_occasions (list)
- price_range (string)
- adventurousness_score (1-10)
- flavor_preferences (list)
- summary (string)
"""
    try:
        response = call_agent("user_profile_generator", user_message)
        user_profile = json.loads(response)
        print(f"  Profile summary: {user_profile.get('summary', 'N/A')}")
    except Exception as e:
        print(f"  Warning: {e}")
        user_profile = {"error": str(e)}

    state["user_profile"] = user_profile
    state["workflow_step"] = "profile_generated"
    return state


def node_retrieve_candidates(state: dict) -> dict:
    print("\n[Phase 2] Retrieving candidates…")

    user_message = f"""Based on this user profile:
{json.dumps(state['user_profile'], indent=2)}

Simulate retrieving top 20 restaurants and top 20 recipes from a vector database.

Return JSON with two arrays:
- restaurants: [{{"name": str, "cuisine": str, "price": str, "rating": float, "description": str}}]
- recipes: [{{"name": str, "cuisine": str, "difficulty": str, "prep_time": str, "description": str}}]

Make the results realistic and diverse.
"""
    try:
        response = call_agent("rag_retriever", user_message)
        data = json.loads(response)
        restaurants = data.get("restaurants", [])
        recipes = data.get("recipes", [])
        print(f"  Retrieved {len(restaurants)} restaurants, {len(recipes)} recipes")
    except Exception as e:
        print(f"  Warning: {e}")
        restaurants, recipes = [], []

    state["retrieved_restaurants"] = restaurants
    state["retrieved_recipes"] = recipes
    state["workflow_step"] = "candidates_retrieved"
    return state


def node_analyze_trends(state: dict) -> dict:
    print("\n[Phase 3a] Analyzing food trends…")

    user_message = f"""Analyze current food trends in these options:

Restaurants: {json.dumps(state['retrieved_restaurants'][:5], indent=2)}
Recipes: {json.dumps(state['retrieved_recipes'][:5], indent=2)}

Identify 3-5 relevant trends.
Return JSON: {{"trends": [{{"name": str, "description": str, "relevance": str}}]}}
"""
    try:
        response = call_agent("food_trend_analyst", user_message)
        trend_analysis = json.loads(response)
        print(f"  Identified {len(trend_analysis.get('trends', []))} trends")
    except Exception as e:
        print(f"  Warning: {e}")
        trend_analysis = {"error": str(e)}

    state["trend_analysis"] = trend_analysis
    return state


def node_analyze_styles(state: dict) -> dict:
    print("\n[Phase 3b] Analyzing food styles…")

    user_message = f"""Analyze the food styles and flavor profiles of these options:

Restaurants: {json.dumps(state['retrieved_restaurants'][:5], indent=2)}
Recipes: {json.dumps(state['retrieved_recipes'][:5], indent=2)}
User Profile: {json.dumps(state['user_profile'], indent=2)}

Return JSON: {{"styles": [{{"name": str, "description": str, "match": str}}]}}
"""
    try:
        response = call_agent("food_style_expert", user_message)
        style_analysis = json.loads(response)
        print("  Style analysis complete")
    except Exception as e:
        print(f"  Warning: {e}")
        style_analysis = {"error": str(e)}

    state["style_analysis"] = style_analysis
    return state


def node_evaluate_nutrition(state: dict) -> dict:
    print("\n[Phase 3c] Evaluating nutrition…")

    user_message = f"""Evaluate the nutritional fit of these options:

User Profile: {json.dumps(state['user_profile'], indent=2)}
Restaurants: {json.dumps(state['retrieved_restaurants'][:5], indent=2)}
Recipes: {json.dumps(state['retrieved_recipes'][:5], indent=2)}

Return JSON: {{"compliant_items": [], "flagged_items": [], "nutritional_highlights": []}}
"""
    try:
        response = call_agent("nutrition_expert", user_message)
        nutrition_analysis = json.loads(response)
        print("  Nutrition evaluation complete")
    except Exception as e:
        print(f"  Warning: {e}")
        nutrition_analysis = {"error": str(e)}

    state["nutrition_analysis"] = nutrition_analysis
    return state


def node_generate_recommendations(state: dict) -> dict:
    print("\n[Phase 4] Generating final recommendations…")

    user_message = f"""Synthesize these insights into top 5 restaurant and top 5 recipe recommendations:

User Profile: {json.dumps(state['user_profile'], indent=2)}
Restaurants: {json.dumps(state['retrieved_restaurants'][:10], indent=2)}
Recipes: {json.dumps(state['retrieved_recipes'][:10], indent=2)}
Trends: {json.dumps(state['trend_analysis'], indent=2)}
Styles: {json.dumps(state['style_analysis'], indent=2)}
Nutrition: {json.dumps(state['nutrition_analysis'], indent=2)}

Return JSON:
{{
  "restaurants": [{{"name": str, "reasoning": str}}],
  "recipes": [{{"name": str, "reasoning": str}}]
}}

Each reasoning: 2-3 sentences explaining why it's a great match.
"""
    try:
        response = call_agent("recommendation_expert", user_message)
        recommendations = json.loads(response)
        print(f"  {len(recommendations.get('restaurants', []))} restaurant recommendations")
        print(f"  {len(recommendations.get('recipes', []))} recipe recommendations")
    except Exception as e:
        print(f"  Warning: {e}")
        recommendations = {"error": str(e)}

    state["final_recommendations"] = recommendations
    state["workflow_step"] = "complete"
    return state


# ---------------------------------------------------------------------------
# Main workflow runner
# ---------------------------------------------------------------------------

def run_workflow(user_input: str) -> dict:
    """
    Execute the full multi-agent pipeline and return the final state.

    Args:
        user_input: Free-text description of the user's preferences.

    Returns:
        State dict containing user_profile, retrieved data, analyses,
        and final_recommendations.
    """
    state: Dict[str, Any] = {
        "user_input": user_input,
        "user_profile": {},
        "retrieved_restaurants": [],
        "retrieved_recipes": [],
        "trend_analysis": {},
        "style_analysis": {},
        "nutrition_analysis": {},
        "final_recommendations": {},
        "workflow_step": "start",
    }

    # Phase 1 – sequential
    state = node_generate_profile(state)

    # Phase 2 – sequential
    state = node_retrieve_candidates(state)

    # Phase 3 – parallel
    print("\n[Phase 3] Running analysis agents in parallel…")
    with ThreadPoolExecutor(max_workers=3) as executor:
        f_trends = executor.submit(node_analyze_trends, dict(state))
        f_styles = executor.submit(node_analyze_styles, dict(state))
        f_nutrition = executor.submit(node_evaluate_nutrition, dict(state))

        state["trend_analysis"] = f_trends.result()["trend_analysis"]
        state["style_analysis"] = f_styles.result()["style_analysis"]
        state["nutrition_analysis"] = f_nutrition.result()["nutrition_analysis"]

    # Phase 4 – sequential
    state = node_generate_recommendations(state)

    return state


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_recommendations(result: Dict[str, Any]):
    print("\n" + "=" * 70)
    print("RECOMMENDATION EVALUATION")
    print("=" * 70)

    profile = result.get("user_profile", {})
    recommendations = result.get("final_recommendations", {})
    restaurants = recommendations.get("restaurants", [])
    recipes = recommendations.get("recipes", [])

    print(f"Restaurant recommendations: {len(restaurants)}")
    print(f"Recipe recommendations:     {len(recipes)}")

    if profile.get("dietary_restrictions"):
        print(f"Dietary restrictions: {', '.join(profile['dietary_restrictions'])}")

    if restaurants:
        first = restaurants[0]
        print(f"\nTop restaurant: {first.get('name', 'N/A')}")
        print(f"  Reasoning: {first.get('reasoning', 'N/A')}")

    if recipes:
        first = recipes[0]
        print(f"\nTop recipe: {first.get('name', 'N/A')}")
        print(f"  Reasoning: {first.get('reasoning', 'N/A')}")

    print("=" * 70)
