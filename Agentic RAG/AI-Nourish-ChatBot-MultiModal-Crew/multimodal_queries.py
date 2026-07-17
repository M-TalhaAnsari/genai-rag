import requests
import os
import base64
from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


url_image_1 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5uo16pKhdB1f2Vz7H8Utkg/image-1.png'
url_image_2 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/fsuegY1q_OxKIxNhf6zeYg/image-2.png'
url_image_3 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/KCh_pM9BVWq_ZdzIBIA9Fw/image-3.png'
url_image_4 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/VaaYLw52RaykwrE3jpFv7g/image-4.png'

image_urls = [url_image_1, url_image_2, url_image_3, url_image_4] 


encoded_images = []

for url in image_urls: 
    encoded_images.append(base64.b64encode(requests.get(url).content).decode("utf-8"))


def generate_model_response(
    encoded_image,
    user_query,
    assistant_prompt="You are a helpful assistant. Answer in 1 or 2 sentences."
):

    image_bytes = base64.b64decode(encoded_image)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            assistant_prompt,
            user_query,
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            )
        ]
    )

    return response.text

user_query = "Describe the photo"

for i, image in enumerate(encoded_images):
    response = generate_model_response(image, user_query)
    print(f"Description for image {i+1}: {response}")