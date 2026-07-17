import logging
import os
import time
from typing import List, Optional

import requests
from crewai.tools import tool
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))


def _load_image_bytes(image_input: str) -> bytes:
    """Load raw image bytes from a local file path or a remote URL."""
    if image_input.startswith("http"):
        response = requests.get(image_input)
        response.raise_for_status()
        return response.content

    if not os.path.exists(image_input):
        raise FileNotFoundError(f"Image file not found: {image_input}")
    with open(image_input, "rb") as f:
        return f.read()


def _guess_mime_type(image_input: str) -> str:
    ext = os.path.splitext(image_input)[1].lower()
    return {
        ".png": "image/png",
        ".webp": "image/webp",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(ext, "image/jpeg")


def _call_with_retry(fn, max_attempts: int = 3, base_delay: int = 25):
    """Retry a Gemini call a few times if it hits a transient rate limit."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) and attempt < max_attempts - 1:
                logger.warning(
                    "Gemini rate limit hit (attempt %s/%s), retrying in %ss...",
                    attempt + 1, max_attempts, base_delay,
                )
                time.sleep(base_delay)
                continue
            raise


def _extract_text(response, context: str) -> str:
    """Pull .text off a Gemini response, raising a clear error if it's empty."""
    if response.text is not None:
        return response.text

    finish_reason = None
    if response.candidates:
        finish_reason = response.candidates[0].finish_reason
    raise RuntimeError(
        f"Gemini returned no text content while {context} "
        f"(finish_reason={finish_reason}). This usually means the image was "
        "blocked by a safety filter, or the model returned an empty response."
    )


class ExtractIngredientsTool:
    @tool
    def extract_ingredients_from_image(image_input: str):
        """
        Extracts ingredients from an image using Google GenAI.

        Args:
            image_input (str): The path to the image file, or a URL.

        Returns:
            str: A comma-separated list of ingredients.
        """
        image_bytes = _load_image_bytes(image_input)
        mime_type = _guess_mime_type(image_input)

        response = _call_with_retry(lambda: client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                "Identify all visible food ingredients. Return only a comma-separated list.",
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
        ))
        return _extract_text(response, "extracting ingredients")


class FilterIngredientsTool:
    @tool("fileter_ingredients")
    def filter_ingredients(raw_ingredients: str):
        """
        Processes the raw ingredient data and filters out non-food items or noise.

        :param raw_ingredients: Raw ingredients as a string.
        :return: A list of cleaned and relevant ingredients.
        """
        ingredients = [
            ingredient.strip().lower()
            for ingredient in raw_ingredients.split(",")
            if ingredient.strip()
        ]
        return ingredients


class DietaryFilterTool:
    @tool("dietary_filter")
    def dietary_filter(ingredients: List[str], dietary_restrictions: Optional[str] = None) -> List[str]:
        """
        Uses an LLM model to filter ingredients based on dietary restrictions.

        :param ingredients: List of ingredients.
        :param dietary_restrictions: Dietary restrictions (e.g., vegan, gluten-free). Defaults to None.
        :return: Filtered list of ingredients that comply with the dietary restrictions.
        """
        if not dietary_restrictions:
            return ingredients

        prompt = f"""
        You are an AI nutritionist specialized in dietary restrictions.
        Given the following list of ingredients: {', '.join(ingredients)},
        and the dietary restriction: {dietary_restrictions},
        remove any ingredient that does not comply with this restriction.
        Return only the compliant ingredients as a comma-separated list with no additional commentary.
        """

        response = _call_with_retry(lambda: client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt],
        ))
        text = _extract_text(response, "filtering ingredients by diet")

        filtered = text.strip().lower()
        return [item.strip() for item in filtered.split(",") if item.strip()]


class NutrientAnalysisTool:
    @tool("Analyze nutritional values and calories of the dish from uploaded image")
    def analyze_image(image_input: str):
        """
        Provide a detailed nutrient breakdown and estimate the total calories of all
        ingredients from the uploaded image.

        :param image_input: the image file path (local) or URL (remote)
        :return: A string with the nutrient breakdown
        """
        image_bytes = _load_image_bytes(image_input)
        mime_type = _guess_mime_type(image_input)

        assistant_prompt = """
        You are an expert nutritionist. Your task is to analyze the food items displayed in the image and provide a detailed nutritional assessment using the following format:

        1. **Identification**: List each identified food item clearly, one per line.

        2. **Portion Size & Calorie Estimation**: For each identified food item, specify the portion size and provide an estimated number of calories. Use bullet points with the following structure:
        - **[Food Item]**: [Portion Size], [Number of Calories] calories

        Example:
        * **Salmon**: 6 ounces, 210 calories
        * **Asparagus**: 3 spears, 25 calories

        3. **Total Calories**: Provide the total number of calories for all food items.

        Example:
        Total Calories: [Number of Calories]

        4. **Nutrient Breakdown**: Include a breakdown of key nutrients such as **Protein**, **Carbohydrates**, **Fats**, **Vitamins**, and **Minerals**. Use bullet points, and for each nutrient provide details about the contribution of each food item.

        Example:
        * **Protein**: Salmon (35g), Asparagus (3g), Tomatoes (1g) = [Total Protein]

        5. **Health Evaluation**: Evaluate the healthiness of the meal in one paragraph.

        6. **Disclaimer**: Include the following exact text as a disclaimer:

        The nutritional information and calorie estimates provided are approximate and are based on general food data.
        Actual values may vary depending on factors such as portion size, specific ingredients, preparation methods, and individual variations.
        For precise dietary advice or medical guidance, consult a qualified nutritionist or healthcare provider.

        Format your response exactly like the template above to ensure consistency.
        """

        response = _call_with_retry(lambda: client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                assistant_prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
        ))
        return _extract_text(response, "analyzing nutrients")