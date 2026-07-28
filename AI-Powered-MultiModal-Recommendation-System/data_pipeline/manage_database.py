"""
data_pipeline/manage_database.py
----------------------------------
Interactive CLI for browsing, adding, editing, and deleting restaurant
records in data/structured_restaurant_data.json.

New entries are described in free text and structured by an LLM before saving.

LLM stack: Groq (primary) → Gemini (fallback).

Run:
    python data_pipeline/manage_database.py
"""

import json
import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel

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


def _invoke_llm(messages):
    """Call Groq; fall back to Gemini on any error."""
    try:
        return groq_llm.invoke(messages)
    except Exception as e:
        print(f"Groq failed ({e}), switching to Gemini…")
        return gemini_llm.invoke(messages)


# ---------------------------------------------------------------------------
# Data schema
# ---------------------------------------------------------------------------

class RestaurantData(BaseModel):
    name: str
    location: str
    type: str
    food_style: str
    rating: float
    price_range: int
    signatures: list
    vibe: str
    environment: str
    shortcomings: list
    item_id: int


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_MSG = """
You are an information extraction assistant.

Extract the restaurant information and return ONLY a valid JSON object with exactly these keys:

{
    "name": "",
    "location": "",
    "type": "",
    "food_style": "",
    "rating": 0.0,
    "price_range": 0,
    "signatures": [],
    "vibe": "",
    "environment": "",
    "shortcomings": []
}

Use "" for missing strings, 0 for missing numbers, [] for missing lists.
Return ONLY JSON.
"""


def _extract_restaurant(paragraph: str, item_id: int) -> dict:
    """Use the LLM with structured output to parse a free-text restaurant description."""
    structured_llm = gemini_llm.with_structured_output(RestaurantData)

    restaurant = structured_llm.invoke([
        SystemMessage(content=EXTRACTION_SYSTEM_MSG),
        HumanMessage(content=paragraph),
    ])
    restaurant.item_id = item_id
    return restaurant.model_dump()


# ---------------------------------------------------------------------------
# JSON file helpers
# ---------------------------------------------------------------------------

FILEPATH = "data/structured_restaurant_data.json"


def _load(file_path: str) -> list:
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(file_path: str, data: list):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def _show_card(record: dict, index: int):
    print(f"\n===== Restaurant #{index} =====")
    for key, value in record.items():
        print(f"  {key}: {value}")
    print("=" * 30)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def manage_restaurants(file_path: str = FILEPATH):
    while True:
        data = _load(file_path)
        print(f"\n🏨 RESTAURANT DATABASE | Records: {len(data)}")
        print("1. Browse All (Names)")
        print("2. View Detailed Record")
        print("3. Add New Restaurant")
        print("4. Edit Restaurant Info")
        print("5. Delete Restaurant")
        print("6. Exit")

        choice = input("\nAction: ").strip()

        if choice == "1":
            print("\n--- Current Listings ---")
            for i, record in enumerate(data):
                print(f"  {i}: {record.get('name', 'N/A')}")

        elif choice == "2":
            try:
                index = int(input("Enter record index: "))
                if 0 <= index < len(data):
                    _show_card(data[index], index)
                else:
                    print("Invalid index.")
            except ValueError:
                print("Please enter a number.")

        elif choice in ("3", "4", "5"):
            print("\n❗ SECURITY WARNING: You are entering write-mode.")
            confirm = input("Are you sure? (type 'yes' to proceed): ").strip().lower()
            if confirm != "yes":
                print("Operation cancelled.")
                continue

            if choice == "3":
                item_id = 1_000_000 + len(data) + 1
                paragraph = input("Enter restaurant description: ")
                new_record = _extract_restaurant(paragraph, item_id)
                data.append(new_record)
                _save(file_path, data)
                print("✅ Restaurant added.")

            elif choice == "4":
                try:
                    index = int(input("Enter record index: "))
                    if 0 <= index < len(data):
                        for key in data[index]:
                            new_val = input(f"New value for '{key}' (Enter to skip): ").strip()
                            if new_val:
                                data[index][key] = new_val
                        _save(file_path, data)
                        print("✅ Record updated.")
                    else:
                        print("Invalid index.")
                except ValueError:
                    print("Please enter a number.")

            elif choice == "5":
                try:
                    index = int(input("Enter record index: "))
                    if 0 <= index < len(data):
                        data.pop(index)
                        _save(file_path, data)
                        print("✅ Record deleted.")
                    else:
                        print("Invalid index.")
                except ValueError:
                    print("Please enter a number.")

        elif choice == "6":
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    manage_restaurants()
