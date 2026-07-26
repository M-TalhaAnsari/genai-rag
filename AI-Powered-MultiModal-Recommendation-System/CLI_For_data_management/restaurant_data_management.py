
import json
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel

FILEPATH = 'data/structured_restaurant_data.json'

from pydantic import BaseModel



def restaurant_data_structure_prompt_generation(restaurant_paragraph):

    system_msg = """
You are an information extraction assistant.

Extract the restaurant information and return ONLY a valid JSON object.

The JSON MUST contain exactly these keys:

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

Do not omit any key.
If a value cannot be determined, use:

"" for strings
0 for numbers
[] for lists

Return ONLY JSON.
"""

    prompt_txt = restaurant_paragraph

    return system_msg, prompt_txt

gemini_llm = ChatGoogleGenerativeAI(
    api_key=os.environ.get("GOOGLE_API_KEY"),
    model="gemini-3.5-flash",
    temperature=0
)

# Use Groq normally
llm = gemini_llm

def new_data_entry_process(paragraph, itemId):

    system_msg, prompt_txt = restaurant_data_structure_prompt_generation(paragraph)

    structured_llm = gemini_llm.with_structured_output(RestaurantData)

    restaurant = structured_llm.invoke([
        SystemMessage(content=system_msg),
        HumanMessage(content=prompt_txt)
    ])

    restaurant.itemId = itemId

    return restaurant.model_dump()
    


def load_data(file_path):
    if not os.path.exists(file_path):
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_data(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def show_restaurant_card(res, index):
    print(f"\n===== Restaurant #{index} =====")
    for key, value in res.items():
        print(f"{key}: {value}")
    print("=" * 30)


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
    itemId: int

def manage_restaurants(file_path):
    while True:
        data = load_data(file_path)
        print(f"\n🏨 RESTAURANT DATABASE | Records: {len(data)}")
        print("1. Browse All (Names)")
        print("2. View Detailed Record")
        print("3. Add New Restaurant")
        print("4. Edit Restaurant Info")
        print("5. Delete Restaurant")
        print("6. Exit")
        
        choice = input("\nAction: ")

        if choice == '1':
            print("\n--- Current Listings ---")
            for key, record in enumerate(data):
                name = record.get("name", "N/A")
                print(f"{key}: {name}")
        
        elif choice == '2': 
            try:
                index = int(input("Enter record index: "))
                if 0 <= index < len(data):
                    show_restaurant_card(data[index], index)
                else:
                    print("Invalid index.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        elif choice in ['3', '4', '5']:
            # Strict Security Warning
            print("\n❗ SECURITY WARNING: You are entering write-mode.")
            print("Changes will be saved to the database immediately.")
            confirm = input("Are you sure? (type 'yes' to proceed): ").lower()
            if confirm != 'yes':
                print("Operation cancelled.")
                continue

            if choice == '3': # ADD NEW DATA
                itemId = 1000000 + len(data) + 1   #the item id for the new data
                paragraph=input("Enter new restaurant description: ")
                new_restaurant = new_data_entry_process(paragraph, itemId=itemId)
                data.append(new_restaurant)
                save_data(file_path, data)
                print("✅ Restaurant added.")

            elif choice == '4': # EDIT DATA
                try:
                    index = int(input("Enter record index: "))
                    if 0 <= index < len(data):
                        for key in data[index].keys():
                            new_value = input(f"Enter new value for {key} (or press Enter to skip): ")
                            if new_value:
                                data[index][key] = new_value
                        save_data(file_path, data)
                        print("✅ Record updated.")
                    else:
                        print("Invalid index.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

            elif choice == '5': # DELETE DATA
                try:
                    index = int(input("Enter record index: "))
                    if 0 <= index < len(data):
                        data.pop(index)
                        save_data(file_path, data)
                        print("✅ Record deleted.")
                    else:
                        print("Invalid index.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

        elif choice == '6': # EXIT
            break
        else:
            print("Invalid input.")


