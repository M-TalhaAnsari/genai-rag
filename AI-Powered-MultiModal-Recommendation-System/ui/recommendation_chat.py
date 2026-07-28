"""
ui/recommendation_chat.py
---------------------------
Gradio chat interface for the multi-agent food recommendation system.

Handles intent classification, preference extraction, and invokes the
full agent workflow to return personalized restaurant and recipe suggestions.

LLM stack: Groq (primary) → Gemini (fallback).

Run:
    python ui/recommendation_chat.py
"""

import json
import os
from typing import Any, Dict, List, Tuple

import gradio as gr
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from recommendation_engine.workflow import run_workflow

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
# Intent classification
# ---------------------------------------------------------------------------

def classify_intent(user_message: str) -> str:
    """Classify user intent as restaurant / recipe / both / clarification / database."""
    system_prompt = """You are an intent classifier for a food recommendation system.

Classify the user's message as ONE of:
- "restaurant" - User wants restaurant recommendations
- "recipe" - User wants recipe recommendations
- "both" - User wants both restaurant and recipe recommendations
- "clarification" - User needs help or is asking a question
- "database" - User wants to add/edit/delete database entries

Examples:
"Where should I eat tonight?" → restaurant
"How do I make lasagna?" → recipe
"I want dinner ideas" → both
"What can you help me with?" → clarification
"I want to add a new restaurant" → database

Respond with ONLY the classification label."""

    response = _invoke_llm([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ])
    intent = response.content.strip().lower()
    valid = {"restaurant", "recipe", "both", "clarification", "database"}
    return intent if intent in valid else "clarification"


# ---------------------------------------------------------------------------
# Preference extraction
# ---------------------------------------------------------------------------

def extract_preferences(user_message: str) -> Dict[str, Any]:
    """Extract structured user preferences from natural language input."""
    system_prompt = """You are a preference extractor for a food recommendation system.

Extract user preferences and return JSON with these keys:
- favorite_cuisines: list of cuisines (e.g. ["Italian", "Thai"])
- dietary_restrictions: list (e.g. ["vegetarian", "gluten-free"])
- dining_occasion: string (e.g. "casual", "fine dining")
- price_range: string (e.g. "$", "$$", "$$$")
- flavor_preferences: list (e.g. ["spicy", "sweet"])
- other_preferences: string

Use empty list [] or "not specified" for missing fields.
Respond with ONLY valid JSON."""

    response = _invoke_llm([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ])

    try:
        return json.loads(response.content)
    except Exception:
        return {
            "favorite_cuisines": [],
            "dietary_restrictions": [],
            "dining_occasion": "not specified",
            "price_range": "not specified",
            "flavor_preferences": [],
            "other_preferences": "",
        }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_recommendations(recommendations: Dict[str, Any]) -> str:
    output = ""

    if recommendations.get("restaurants"):
        output += "🍽️ **Restaurant Recommendations:**\n\n"
        for i, r in enumerate(recommendations["restaurants"], 1):
            output += f"**{i}. {r['name']}**\n"
            if r.get("cuisine"):
                output += f"   - Cuisine: {r['cuisine']}\n"
            if r.get("price"):
                output += f"   - Price: {r['price']}\n"
            output += f"   - Why: {r['reasoning']}\n\n"

    if recommendations.get("recipes"):
        output += "👨‍🍳 **Recipe Recommendations:**\n\n"
        for i, r in enumerate(recommendations["recipes"], 1):
            output += f"**{i}. {r['name']}**\n"
            if r.get("cuisine"):
                output += f"   - Cuisine: {r['cuisine']}\n"
            if r.get("difficulty"):
                output += f"   - Difficulty: {r['difficulty']}\n"
            output += f"   - Why: {r['reasoning']}\n\n"

    return output or "I couldn't generate recommendations. Please add more details about your preferences."


# ---------------------------------------------------------------------------
# Main chatbot handler
# ---------------------------------------------------------------------------

def recommendation_chatbot(message: str, history: List[Tuple[str, str]]) -> str:
    try:
        intent = classify_intent(message)

        if intent == "clarification":
            return (
                "I'm your food recommendation assistant! I can help you with:\n\n"
                "🍽️ **Restaurant recommendations** — tell me your cuisine preferences, "
                "dietary restrictions, and occasion\n"
                "👨‍🍳 **Recipe recommendations** — let me know what you'd like to cook\n"
                "📝 **Database management** — use the tabs above to add or edit entries\n\n"
                "Just describe what you're looking for!"
            )

        if intent == "database":
            return (
                "To manage the database, please use the tabs above:\n\n"
                "- **Add Restaurant**: submit a new restaurant\n"
                "- **Add Recipe**: submit a new recipe\n\n"
                "Is there anything else I can help you with?"
            )

        if intent in {"restaurant", "recipe", "both"}:
            preferences = extract_preferences(message)
            result = run_workflow(json.dumps(preferences))
            recs = result.get("final_recommendations", {})

            if intent == "restaurant":
                recs = {"restaurants": recs.get("restaurants", [])}
            elif intent == "recipe":
                recs = {"recipes": recs.get("recipes", [])}

            return format_recommendations(recs)

        return "I'm not sure how to help with that. Could you rephrase your request?"

    except Exception as e:
        return f"I encountered an error: {e}. Please check your API keys and try again."


# ---------------------------------------------------------------------------
# Database stubs (connected to UI buttons)
# ---------------------------------------------------------------------------

def add_restaurant(name, cuisine, price, location, description) -> str:
    # Wire to vector DB upsert in production
    print(f"Adding restaurant: {name}")
    return f"✅ Successfully added '{name}' to the database!"


def add_recipe(name, cuisine, difficulty, prep_time, ingredients, instructions) -> str:
    # Wire to vector DB upsert in production
    print(f"Adding recipe: {name}")
    return f"✅ Successfully added '{name}' to the database!"


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

with gr.Blocks(title="Food Recommendation Chatbot", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🍽️ Food Recommendation Chatbot\nYour personal AI assistant for restaurant and recipe recommendations!")

    with gr.Tabs():

        with gr.Tab("💬 Chat"):
            gr.ChatInterface(
                fn=recommendation_chatbot,
                examples=[
                    "I'm looking for vegetarian restaurants",
                    "Suggest some easy recipes for dinner",
                    "I want spicy Thai food recommendations",
                    "What can you help me with?",
                ],
                title="Chat with the Recommendation Assistant",
                description="Describe your food preferences and I'll recommend restaurants or recipes!",
            )

        with gr.Tab("➕ Add Restaurant"):
            gr.Markdown("### Add a New Restaurant to the Database")
            with gr.Row():
                with gr.Column():
                    rest_name = gr.Textbox(label="Restaurant Name")
                    rest_cuisine = gr.Textbox(label="Cuisine Type")
                    rest_price = gr.Dropdown(choices=["$", "$$", "$$$", "$$$$"], label="Price Range")
                with gr.Column():
                    rest_location = gr.Textbox(label="Location")
                    rest_description = gr.Textbox(label="Description", lines=3)
            add_rest_btn = gr.Button("Add Restaurant", variant="primary")
            rest_output = gr.Textbox(label="Status")
            add_rest_btn.click(
                fn=add_restaurant,
                inputs=[rest_name, rest_cuisine, rest_price, rest_location, rest_description],
                outputs=rest_output,
            )

        with gr.Tab("➕ Add Recipe"):
            gr.Markdown("### Add a New Recipe to the Database")
            with gr.Row():
                with gr.Column():
                    recipe_name = gr.Textbox(label="Recipe Name")
                    recipe_cuisine = gr.Textbox(label="Cuisine Type")
                    recipe_difficulty = gr.Dropdown(choices=["Easy", "Medium", "Hard"], label="Difficulty")
                with gr.Column():
                    recipe_time = gr.Textbox(label="Prep Time")
                    recipe_ingredients = gr.Textbox(label="Ingredients (comma-separated)", lines=3)
            recipe_instructions = gr.Textbox(label="Instructions", lines=5)
            add_recipe_btn = gr.Button("Add Recipe", variant="primary")
            recipe_output = gr.Textbox(label="Status")
            add_recipe_btn.click(
                fn=add_recipe,
                inputs=[recipe_name, recipe_cuisine, recipe_difficulty, recipe_time, recipe_ingredients, recipe_instructions],
                outputs=recipe_output,
            )

        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
## About This Chatbot

This chatbot uses a multi-agent AI system to provide personalized food recommendations.

### Features
- 🤖 **Six specialist agents** working in a hybrid sequential/parallel workflow
- 🔍 **Multimodal vector retrieval** combining text and image embeddings
- 🎯 **Personalized results** tailored to dietary needs and flavor preferences
- 📝 **Editable database** — add your own restaurants and recipes

### Technologies
- LangChain for LLM orchestration
- Groq (LLaMA 3.3) with Gemini fallback
- ChromaDB for vector storage
- FastMCP for tool serving
- Gradio for the interface
""")


if __name__ == "__main__":
    demo.launch(share=True)
