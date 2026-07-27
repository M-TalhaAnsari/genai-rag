import os
import json
from typing import TypedDict, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

import agents

agents_config = { "user_profile_generator": agents.user_profile_agent_config, "rag_retriever": agents.rag_retriever_agent_config, "food_trend_analyst": agents.food_trend_analyst_config, "food_style_expert": agents.food_style_expert_config, "nutrition_expert": agents.nutrition_expert_config, "recommendation_expert":    agents.recommendation_expert_config}

# DEFINED THE SHARED STATE STRUCTURE

# defined the shared state structure
# everynode reads from and write to this state
INITIAL_STATE = {
    "user_input":"",

    # phase 1
    "user_profile": {},

    # phase 2
    "retrieved_restaurants": [],
    "retrieved_recipes": [],

    # phase 3
    "trend_analysis": {},
    "style_analysis": {},
    "nutrition_analysis": {},

    # phase 4
    "final_recommendations": [],

    # metadata
    "workflow_step":"start"

}

# CREATE THE NODE FUNCTIONS
"""
1. Receives the current task
2. Performs the agents task
3. returns an upadated state

"""

# HELPER FUNCTION TO RUN AN AGENT
def call_agent(agent_key: str, user_message: str) -> str:
    """Call an agent with a specific message and return its response."""
    config = agents_config[agent_key]
    system_prompt = f"""You are a {config['role']}.
    
Your goal: {config['goal']}

Your background: {config['backstory']}

Respond with structured, actionable output."""
    
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return response.choices[0].message.content



# NODE 1 GENERATE THE USER PROFILE
def node_generate_profile(state: dict) -> dict:
    """Generate user profile from input data."""
    print("\n[Phase 1] Generating user profile...")
    
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
        print(f"✓ User profile generated: {user_profile.get('summary', 'No summary')}")
    except Exception as e:
        print(f"⚠ Error generating profile: {e}")
        user_profile = {"error": str(e)}
    
    state["user_profile"] = user_profile
    state["workflow_step"] = "profile_generated"
    return state

# NODE 2 RETRIEVE THE CANDIDATE
def node_retrieve_candidates(state: dict) -> dict:
    """Retrieve restaurant and recipe candidates from vector database."""
    print("\n[Phase 2] Retrieving candidates from vector database...")
    
    profile = state["user_profile"]
    
    user_message = f"""Based on this user profile:
{json.dumps(profile, indent=2)}

Simulate retrieving top 20 restaurants and top 20 recipes from a vector database.

Return JSON with two arrays:
- restaurants: [{{"name": str, "cuisine": str, "price": str, "rating": float, "description": str}}]
- recipes: [{{"name": str, "cuisine": str, "difficulty": str, "prep_time": str, "description": str}}]

Make the results realistic and diverse.
"""
    
    try:
        response = call_agent("rag_retriever", user_message)
        retrieved_data = json.loads(response)
        restaurants = retrieved_data.get("restaurants", [])
        recipes = retrieved_data.get("recipes", [])
        print(f"✓ Retrieved {len(restaurants)} restaurants and {len(recipes)} recipes")
    except Exception as e:
        print(f"⚠ Error retrieving candidates: {e}")
        restaurants, recipes = [], []
    
    state["retrieved_restaurants"] = restaurants
    state["retrieved_recipes"] = recipes
    state["workflow_step"] = "candidates_retrieved"
    return state

# NODE 3 ANALYZE TREND
def node_analyze_trends(state: dict) -> dict:
    """Analyze food trends in the retrieved candidates."""
    print("\n[Phase 3a] Analyzing food trends...")
    
    restaurants = state["retrieved_restaurants"]
    recipes = state["retrieved_recipes"]
    
    user_message = f"""Analyze current food trends in these options:

Restaurants: {json.dumps(restaurants[:5], indent=2)}
Recipes: {json.dumps(recipes[:5], indent=2)}

Identify 3-5 relevant trends and explain how they align with modern dining culture.
Return JSON: {{"trends": [{{"name": str, "description": str, "relevance": str}}]}}
"""
    
    try:
        response = call_agent("food_trend_analyst", user_message)
        trend_analysis = json.loads(response)
        print(f"✓ Identified {len(trend_analysis.get('trends', []))} trends")
    except Exception as e:
        print(f"⚠ Error analyzing trends: {e}")
        trend_analysis = {"error": str(e)}
    
    state["trend_analysis"] = trend_analysis
    return state

# ANALYZE FOOD STYLES
def node_analyze_styles(state: dict) -> dict:
    """Analyze food styles and flavor profiles."""
    print("\n[Phase 3b] Analyzing food styles...")
    
    restaurants = state["retrieved_restaurants"]
    recipes = state["retrieved_recipes"]
    profile = state["user_profile"]
    
    user_message = f"""Analyze the food styles and flavor profiles of these options:

Restaurants: {json.dumps(restaurants[:5], indent=2)}
Recipes: {json.dumps(recipes[:5], indent=2)}
User Profile: {json.dumps(profile, indent=2)}

Identify the dominant food styles and flavor profiles, and explain how they match the user's preferences.
Return JSON: {{"styles": [{{"name": str, "description": str, "match": str}}]}}
"""
    
    try:
        response = call_agent("food_style_expert", user_message)
        style_analysis = json.loads(response)
        print(f"✓ Style analysis completed")
    except Exception as e:
        print(f"⚠ Error analyzing styles: {e}")
        style_analysis = {"error": str(e)}
    
    state["style_analysis"] = style_analysis
    return state

# NODE 5: EVALUATE NUTRITIONS
def node_evaluate_nutrition(state: dict) -> dict:
    """Evaluate nutritional aspects and dietary compliance."""
    print("\n[Phase 3c] Evaluating nutrition...")
    
    restaurants = state["retrieved_restaurants"]
    recipes = state["retrieved_recipes"]
    profile = state["user_profile"]
    
    user_message = f"""Evaluate the nutritional fit of these options:

User Profile: {json.dumps(profile, indent=2)}
Restaurants: {json.dumps(restaurants[:5], indent=2)}
Recipes: {json.dumps(recipes[:5], indent=2)}

Check dietary restrictions, allergens, and nutritional balance.
Return JSON: {{"compliant_items": [], "flagged_items": [], "nutritional_highlights": []}}
"""
    
    try:
        response = call_agent("nutrition_expert", user_message)
        nutrition_analysis = json.loads(response)
        print(f"✓ Nutrition evaluation completed")
    except Exception as e:
        print(f"⚠ Error evaluating nutrition: {e}")
        nutrition_analysis = {"error": str(e)}
    
    state["nutrition_analysis"] = nutrition_analysis
    return state

# NODE 6 GENERATE RECOMMENDATIONS
def node_generate_recommendations(state: dict) -> dict:
    """Synthesize all analyses into final recommendations."""
    print("\n[Phase 4] Generating final recommendations...")
    
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

Each reasoning should be 2-3 sentences explaining why it's a great match.
"""
    
    try:
        response = call_agent("recommendation_expert", user_message)
        recommendations = json.loads(response)
        print(f"✓ Generated {len(recommendations.get('restaurants', []))} restaurant recommendations")
        print(f"✓ Generated {len(recommendations.get('recipes', []))} recipe recommendations")
    except Exception as e:
        print(f"⚠ Error generating recommendations: {e}")
        recommendations = {"error": str(e)}
    
    state["final_recommendations"] = recommendations
    state["workflow_step"] = "complete"
    return state

# WORKFLOW GRAPH
def run_workflow(user_input: str) -> dict:
    """Run the full multi-agent workflow.
    
    Phases:
      1. User Analysis      (sequential)
      2. Data Retrieval      (sequential)
      3. Analysis            (parallel – trends, styles, nutrition)
      4. Synthesis           (sequential)
    """
    
    # Initialize shared state
    state = {
        "user_input": user_input,
        "user_profile": {},
        "retrieved_restaurants": [],
        "retrieved_recipes": [],
        "trend_analysis": {},
        "style_analysis": {},
        "nutrition_analysis": {},
        "final_recommendations": {},
        "workflow_step": "start"
    }
    
    # Phase 1 – Sequential
    state = node_generate_profile(state)
    
    # Phase 2 – Sequential
    state = node_retrieve_candidates(state)
    
    # Phase 3 – Parallel using ThreadPoolExecutor
    print("\n[Phase 3] Running analysis agents in parallel...")
    
    # Each function needs its own copy of state to read from,
    # and we merge their outputs back afterwards.
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_trends   = executor.submit(node_analyze_trends, dict(state))
        future_styles   = executor.submit(node_analyze_styles, dict(state))
        future_nutrition = executor.submit(node_evaluate_nutrition, dict(state))
        
        result_trends   = future_trends.result()
        result_styles   = future_styles.result()
        result_nutrition = future_nutrition.result()
        
    # Merge parallel results back into state
    state["trend_analysis"]    = result_trends["trend_analysis"]
    state["style_analysis"]    = result_styles["style_analysis"]
    state["nutrition_analysis"] = result_nutrition["nutrition_analysis"]
    
    # Phase 4 – Sequential
    state = node_generate_recommendations(state)
    
    return state


def evaluate_recommendations(result: Dict[str, Any]):
    """Evaluate the quality of recommendations."""
    print("\n" + "="*80)
    print("RECOMMENDATION EVALUATION")
    print("="*80)
    
    profile = result.get("user_profile", {})
    recommendations = result.get("final_recommendations", {})
    
    # Check if recommendations exist
    restaurants = recommendations.get("restaurants", [])
    recipes = recommendations.get("recipes", [])
    
    print(f"\n✓ Number of restaurant recommendations: {len(restaurants)}")
    print(f"✓ Number of recipe recommendations: {len(recipes)}")
    
    # Check dietary compliance
    dietary_restrictions = profile.get("dietary_restrictions", [])
    if dietary_restrictions:
        print(f"\n✓ Dietary restrictions identified: {', '.join(dietary_restrictions)}")
        print("  → Check if recommendations respect these restrictions")
    
    # Check diversity
    favorite_cuisines = profile.get("favorite_cuisines", [])
    if favorite_cuisines:
        print(f"\n✓ Favorite cuisines: {', '.join(favorite_cuisines)}")
        print("  → Check if recommendations include these cuisines")
    
    # Evaluate reasoning quality
    if restaurants:
        print(f"\n✓ First restaurant recommendation:")
        print(f"  Name: {restaurants[0].get('name', 'N/A')}")
        print(f"  Reasoning: {restaurants[0].get('reasoning', 'N/A')}")
    
    if recipes:
        print(f"\n✓ First recipe recommendation:")
        print(f"  Name: {recipes[0].get('name', 'N/A')}")
        print(f"  Reasoning: {recipes[0].get('reasoning', 'N/A')}")
    
    print("\n" + "="*80)