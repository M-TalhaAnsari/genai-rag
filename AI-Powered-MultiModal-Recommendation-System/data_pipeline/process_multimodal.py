"""
data_pipeline/process_multimodal.py
-------------------------------------
Augments recipe and user-review data with image captions produced by a
vision-capable LLM.

LLM stack: Gemini Vision (primary) → Groq text fallback for caption repair.

Run this once after collecting raw data. Outputs:
  data/augmented_food_recipe.json
  data/augmented_user_review.json
"""

import ast
import base64
import json
import os
import warnings

import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

warnings.filterwarnings("ignore")
load_dotenv()

# ---------------------------------------------------------------------------
# LLM setup – Gemini Vision primary (supports images), Groq text fallback
# ---------------------------------------------------------------------------

gemini_llm = ChatGoogleGenerativeAI(
    api_key=os.environ.get("GOOGLE_API_KEY"),
    model="gemini-1.5-flash",
    temperature=0,
)

groq_llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0,
)


def invoke_vision_llm(system_msg: str, prompt_txt: str, image_path: str) -> str:
    """
    Call Gemini with an image + text prompt.
    Falls back to Groq (text-only) if the image call fails.
    """
    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        ext = os.path.splitext(image_path)[-1].lower().lstrip(".")
        mime = f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext}"

        response = gemini_llm.invoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_data}"}},
                {"type": "text", "text": prompt_txt},
            ]),
        ])
        return response.content

    except Exception as e:
        print(f"  Gemini vision failed ({e}), falling back to Groq text-only…")
        response = groq_llm.invoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=f"[Image unavailable] {prompt_txt}"),
        ])
        return response.content


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

def recipe_caption_prompt(food_name: str):
    system_msg = (
        "You are a careful food image captioning assistant. "
        "Describe only what is clearly visible in the food image. "
        "Focus on the named dish, visible ingredients, cooking style, presentation, "
        "portion size, texture, and color. "
        "Do not guess or invent ingredients, recipe steps, cuisine, taste, or nutrition. "
        "Keep the caption concise, factual, and suitable for JSON."
    )
    prompt_txt = (
        f"Generate a concise caption for the food image of '{food_name}'. "
        "Use only visible visual details. "
        "If something is unclear, do not speculate."
    )
    return system_msg, prompt_txt


def review_caption_prompt(review_text: str):
    system_msg = (
        "You are a culinary expert analyzing food images. "
        "Use the user review as context, but only describe what is supported by the image and the review. "
        "Focus on visible food items, presentation, ingredients, and style. "
        "Keep the caption concise and avoid unsupported guesses."
    )
    prompt_txt = (
        f"Based on the following review context, generate a concise description of the food image: "
        f"{review_text}. "
        "Make sure the caption matches the image and reflects the review details when relevant."
    )
    return system_msg, prompt_txt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_image_bytes(url: str) -> bytes:
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return response.content


# ---------------------------------------------------------------------------
# Augmentation functions
# ---------------------------------------------------------------------------

def augment_recipes(recipe_path: str, image_dir: str, output_path: str):
    with open(recipe_path, "r") as f:
        recipes = json.load(f)

    for i, recipe in enumerate(recipes):
        food_name = recipe.get("name", recipe.get("title", "Unknown Dish"))
        image_path = os.path.join(image_dir, f"recipe{i + 1}.png")

        system_msg, prompt_txt = recipe_caption_prompt(food_name)
        recipes[i]["image_description"] = invoke_vision_llm(system_msg, prompt_txt, image_path)

        if (i + 1) % 20 == 0:
            print(f"  Recipes: {i + 1}/{len(recipes)} done")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(recipes, f, indent=4)
    print(f"Saved augmented recipes → {output_path}")


def augment_user_reviews(review_path: str, output_path: str):
    with open(review_path, "r") as f:
        reviews = json.load(f)

    # Normalise image field: stored as string repr of list in some datasets
    for i, review in enumerate(reviews):
        img_field = review.get("image", [])
        if isinstance(img_field, str):
            try:
                reviews[i]["image"] = ast.literal_eval(img_field)
            except (ValueError, SyntaxError):
                reviews[i]["image"] = []

    for i, review in enumerate(reviews):
        image_urls = review.get("images", [])
        captions = []

        for url in image_urls:
            try:
                image_bytes = fetch_image_bytes(url)
                tmp_path = "tmp_review_image.jpg"
                with open(tmp_path, "wb") as f:
                    f.write(image_bytes)

                system_msg, prompt_txt = review_caption_prompt(review.get("review", ""))
                caption = invoke_vision_llm(system_msg, prompt_txt, tmp_path)
                captions.append(caption)
            except Exception as e:
                print(f"  Skipping image {url}: {e}")

        reviews[i]["image_description"] = captions

        if (i + 1) % 20 == 0:
            print(f"  Reviews: {i + 1}/{len(reviews)} done")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(reviews, f, indent=4)
    print(f"Saved augmented reviews → {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    augment_recipes(
        recipe_path="data/food_recipes.json",
        image_dir="data/recipe_images",
        output_path="data/augmented_food_recipe.json",
    )
    augment_user_reviews(
        review_path="data/synthetic_user_reviews.json",
        output_path="data/augmented_user_review.json",
    )
