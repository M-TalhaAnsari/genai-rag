# Required libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# Primary LLM: Groq
groq_llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0
)

# Fallback LLM: Gemini
gemini_llm = ChatGoogleGenerativeAI(
    api_key=os.environ.get("GOOGLE_API_KEY"),
    model="gemini-3.5-flash",
    temperature=0
)

# Use Groq normally
llm = groq_llm

# Load data
file_path = r"AI-Powered-MultiModal-Recommendation-System\data\California-Culinary-Map.txt"

with open(file_path, 'r') as file:
    data = file.read()

# split the restaranut paragraph into a python list 
restaurant_list = data.split('\n\n')

restaurant_list = restaurant_list[1:]  # First item is name

# Define the LLM
def invoke_llm(messages):
    try:
        return groq_llm.invoke(messages)
    
    except Exception as e:
        print("Groq failed, switching to Gemini...")
        print(e)

        return gemini_llm.invoke(messages)

# Prompt Engineering
EXAMPLE_RESTAURANT_PARAGRAPH = restaurant_list[1]  # use the second restaurant paragraph as the example

EXAMPLE_OUTPUT = """
{
    "name": "Mar de Cortez",
    "location": "Santa Monica",
    "type": "casual taqueria",
    "food_style": "Baja-style seafood",
    "rating": 4.2,
    "price_range": 1,
    "signatures": [
        "beer-battered snapper tacos",
        "zesty octopus ceviche"
    ],
    "vibe": "salt-air energy",
    "environment": "a premier sun-drenched spot for open-air dining near the pier.",
    "shortcomings": []
}
"""

def restaurant_data_structure_prompt_generation(restaurant_paragraph):
    base_system_msg = """
You are an expert information extraction assistant.

Your task is to extract restaurant information from a restaurant description
and return ONLY a valid JSON object.

Use the following schema:

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
- Return ONLY JSON.
- Do not include explanations or markdown.
- Convert the number of '$' symbols into an integer.
  Example:
  $ -> 1
  $$ -> 2
  $$$ -> 3
  $$$$ -> 4
- If no shortcomings are mentioned, return an empty list [].
- If a field is missing, use an empty string "" or an empty list [] where appropriate.
"""

    base_user_prompt = f"""
Task:
Extract the restaurant information into the required JSON format.

Restaurant description:
{restaurant_paragraph}

Example:

Input Restaurant Description:
{EXAMPLE_RESTAURANT_PARAGRAPH}

Output:
{EXAMPLE_OUTPUT}
"""

    return base_system_msg, base_user_prompt


# Validate the LLM outputs
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


# Structure all the restaurant data 
def JSON_auto_repair_prompts(candidate_json_output, error_message):
    auto_repair_system_msg = """
You are an expert JSON repair assistant.

Your only task is to repair invalid JSON so that it conforms to the required
schema while preserving the original information whenever possible.

Rules:
- Return ONLY valid JSON.
- Do not include explanations, markdown, or comments.
- Preserve the original meaning and values whenever possible.
- Fix syntax errors, missing commas, quotes, brackets, and invalid data types.
- Use the validation error message as guidance for the necessary corrections.
"""

    auto_repair_prompt = f"""
The following JSON failed validation.

Candidate JSON:
{candidate_json_output}

Validation Error:
{error_message}

Repair the JSON so that it is valid and satisfies the required schema.

Return ONLY the corrected JSON.
"""

    return auto_repair_system_msg, auto_repair_prompt

# Run the loop to go over all the restaurant data in the list
structured_restaurant_list = []

for i, restaurant_paragraph in enumerate(restaurant_list):

    system_msg, user_prompt = restaurant_data_structure_prompt_generation(
        restaurant_paragraph
    )

    candidate_json_output = invoke_llm([
        SystemMessage(content=system_msg),
        HumanMessage(content=user_prompt)
    ]).content


    while True:
        try:
            candidate_json = json.loads(candidate_json_output)

            validated_output = Restaurant.model_validate(candidate_json)

            break

        except Exception as e:

            repair_system_msg, repair_prompt = JSON_auto_repair_prompts(
                candidate_json_output,
                str(e)
            )

            candidate_json_output = invoke_llm([
                SystemMessage(content=repair_system_msg),
                HumanMessage(content=repair_prompt)
            ]).content


    # Only append after successful validation
    structured_restaurant_list.append(
        validated_output.model_dump()
    )


    if (i + 1) % 20 == 0:
        print(f"{i+1} out of {len(restaurant_list)} is done")


print("Done")


print(structured_restaurant_list[5])  


# Save the list to Json
structured_restaurant_list_json = [json.load(response) for response in structured_restaurant_list]

# Asign itemid
for i,response in enumerate(structured_restaurant_list_json):
    response['itemid'] = 1000001+i
    structured_restaurant_list_json[i] = response

filename = "Structured_restaurant_data.json" 

with open(filename, 'w', encoding="utf-8") as f:
    json.dump(structured_restaurant_list_json, f, indent=4)