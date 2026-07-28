"""
data_pipeline/ingest_restaurants.py
-------------------------------------
Reads the raw California-Culinary-Map.txt, structures each restaurant
paragraph into validated JSON using an LLM, and saves the result to
data/structured_restaurant_data.json.

LLM stack: Groq (primary) → Gemini (fallback).
"""

import json
import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

# ---------------------------------------------------------------------------
# LLM setup – Groq primary, Gemini fallback
# ---------------------------------------------------------------------------

groq_llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0,
)

gemini_llm = ChatGoogleGenerativeAI(
    api_key=os.environ.get("GOOGLE_API_KEY"),
    model="gemini-1.5-flash",
    temperature=0,
)


def invoke_llm(messages):
    """Call Groq; fall back to Gemini on any error."""
    try:
        return groq_llm.invoke(messages)
    except Exception as e:
        print(f"Groq failed ({e}), switching to Gemini…")
        return gemini_llm.invoke(messages)


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------

class Restaurant(BaseModel):
    name: str
    location: str
    type: str
    food_style: str
    rating: Optional[float] = Field(default=None, ge=0.0, le=5.0)
    price_range: Optional[int] = None
    signatures: List[str] = Field(default_factory=list)
    vibe: Optional[str] = None
    environment: str
    shortcomings: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

EXAMPLE_PARAGRAPH = ""  # filled at runtime from the second entry in the list

EXAMPLE_OUTPUT = """
{
    "name": "Mar de Cortez",
    "location": "Santa Monica",
    "type": "casual taqueria",
    "food_style": "Baja-style seafood",
    "rating": 4.2,
    "price_range": 1,
    "signatures": ["beer-battered snapper tacos", "zesty octopus ceviche"],
    "vibe": "salt-air energy",
    "environment": "a premier sun-drenched spot for open-air dining near the pier.",
    "shortcomings": []
}
"""


def build_extraction_prompt(paragraph: str, example_paragraph: str):
    system_msg = """
You are an expert information extraction assistant.

Extract restaurant information from the description and return ONLY a valid JSON object.

Schema:
{
    "name": string,
    "location": string,
    "type": string,
    "food_style": string,
    "rating": float,
    "price_range": integer,
    "signatures": list of strings,
    "vibe": string,
    "environment": string,
    "shortcomings": list of strings
}

Rules:
- Return ONLY JSON, no markdown, no explanations.
- Convert '$' symbols to integer: $ → 1, $$ → 2, $$$ → 3, $$$$ → 4.
- Use [] for missing lists and "" for missing strings.
"""
    user_prompt = f"""
Extract restaurant information into JSON.

Restaurant description:
{paragraph}

Example input:
{example_paragraph}

Example output:
{EXAMPLE_OUTPUT}
"""
    return system_msg, user_prompt


def build_repair_prompt(bad_json: str, error: str):
    system_msg = """
You are an expert JSON repair assistant.
Return ONLY valid, corrected JSON. No explanations, no markdown.
Preserve all original values; fix only syntax or type errors.
"""
    user_prompt = f"""
The following JSON failed validation.

JSON:
{bad_json}

Error:
{error}

Return the corrected JSON.
"""
    return system_msg, user_prompt


# ---------------------------------------------------------------------------
# Core extraction loop
# ---------------------------------------------------------------------------

def extract_and_validate(paragraph: str, example_paragraph: str) -> dict:
    """Extract one restaurant paragraph → validated dict, auto-repairing if needed."""
    system_msg, user_prompt = build_extraction_prompt(paragraph, example_paragraph)
    candidate = invoke_llm([
        SystemMessage(content=system_msg),
        HumanMessage(content=user_prompt),
    ]).content

    while True:
        try:
            data = json.loads(candidate)
            return Restaurant.model_validate(data).model_dump()
        except Exception as e:
            repair_sys, repair_usr = build_repair_prompt(candidate, str(e))
            candidate = invoke_llm([
                SystemMessage(content=repair_sys),
                HumanMessage(content=repair_usr),
            ]).content


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(raw_text_path: str, output_path: str):
    with open(raw_text_path, "r", encoding="utf-8") as f:
        raw = f.read()

    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    paragraphs = paragraphs[1:]          # skip header line
    example_paragraph = paragraphs[1]   # used in few-shot prompt

    results = []
    for i, paragraph in enumerate(paragraphs):
        record = extract_and_validate(paragraph, example_paragraph)
        record["item_id"] = 1_000_001 + i
        results.append(record)

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(paragraphs)} processed")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Saved {len(results)} restaurants → {output_path}")


if __name__ == "__main__":
    run(
        raw_text_path="data/California-Culinary-Map.txt",
        output_path="data/structured_restaurant_data.json",
    )
