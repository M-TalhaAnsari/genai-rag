import requests
import re
import base64
import os
import io
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch

model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

processor = AutoProcessor.from_pretrained(model_name)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto"
)
app = Flask(__name__)

def input_image_setup(uploaded_file):
    """
    Encodes the uploaded image file into a base64 string to be used with AI models.

    Parameters:
    - uploaded_file: File-like object uploaded via a file uploader (Streamlit or other frameworks)

    Returns:
    - encoded_image (str): Base64 encoded string of the image data
    """
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        encoded_image = base64.b64encode(bytes_data).decode("utf-8")
        return encoded_image

    else:
       raise FileNotFoundError("No File Uploaded")
    
def format_response(response_text):
    """
    Formats the model response to display each item on a new line as a list.
    Converts numbered items into HTML `<ul>` and `<li>` format.
    Adds additional HTML elements for better presentation of headings and separate sections.
    """
    # Replace section headers that are bolded with '**' to HTML paragraph tags with bold text
    response_text = re.sub(r"\*\*(.*?)\*\*", r"<p><strong>\1</strong></p>", response_text)

    # Convert bullet points denoted by "*" to HTML list items
    response_text = re.sub(r"(?m)^\s*\*\s(.*)", r"<li>\1</li>", response_text)

    # Wrap list items within <ul> tags for proper HTML structure and indentation
    response_text = re.sub(r"(<li>.*?</li>)+", lambda match: f"<ul>{match.group(0)}</ul>", response_text, flags=re.DOTALL)

    # Ensure that all paragraphs have a line break after them for better separation
    response_text = re.sub(r"</p>(?=<p>)", r"</p><br>", response_text)

    # Ensure the disclaimer and other distinct paragraphs have proper line breaks
    response_text = re.sub(r"(\n|\\n)+", r"<br>", response_text)

    return response_text   


def generate_model_response(encoded_image, user_query, assistant_prompt):

    # Decode Base64 image
    image = Image.open(
        io.BytesIO(base64.b64decode(encoded_image))
    ).convert("RGB")

    prompt = assistant_prompt + "\n\n" + user_query

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # Convert chat to model input
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt"
    ).to(model.device)

    # Generate response
    output = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.2,
        top_p=0.5
    )

    response = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return format_response(response)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Retrieve user inputs
        user_query = request.form.get("user_query")
        uploaded_file = request.files.get("file")

        if uploaded_file:
            # Process the uploaded image
            encoded_image = input_image_setup(uploaded_file)

            if not encoded_image:
                flash("Error processing the image. Please try again.", "danger")
                return redirect(url_for("index"))

            # Assistant prompt (can be customized)
            assistant_prompt = """
            You are an expert nutritionist. Your task is to analyze the food items displayed in the image and provide a detailed nutritional assessment using the following format:

        1. **Identification**: List each identified food item clearly, one per line.
        2. **Portion Size & Calorie Estimation**: For each identified food item, specify the portion size and provide an estimated number of calories. Use bullet points with the following structure:
        - **[Food Item]**: [Portion Size], [Number of Calories] calories

        Example:
        *   **Salmon**: 6 ounces, 210 calories
        *   **Asparagus**: 3 spears, 25 calories

        3. **Total Calories**: Provide the total number of calories for all food items.

        Example:
        Total Calories: [Number of Calories]

        4. **Nutrient Breakdown**: Include a breakdown of key nutrients such as **Protein**, **Carbohydrates**, **Fats**, **Vitamins**, and **Minerals**. Use bullet points, and for each nutrient provide details about the contribution of each food item.

        Example:
        *   **Protein**: Salmon (35g), Asparagus (3g), Tomatoes (1g) = [Total Protein]

        5. **Health Evaluation**: Evaluate the healthiness of the meal in one paragraph.

        6. **Disclaimer**: Include the following exact text as a disclaimer:

        The nutritional information and calorie estimates provided are approximate and are based on general food data. 
        Actual values may vary depending on factors such as portion size, specific ingredients, preparation methods, and individual variations. 
        For precise dietary advice or medical guidance, consult a qualified nutritionist or healthcare provider.

        Format your response exactly like the template above to ensure consistency.

        """

            # Generate the model's response
            response = generate_model_response(encoded_image, user_query, assistant_prompt)

            # Render the result
            return render_template("index.htm", user_query=user_query, response=response)

        else:
            flash("Please upload an image file.", "danger")
            return redirect(url_for("index"))

    return render_template("index.htm")

if __name__ == "__main__":
    app.run(debug=True)
