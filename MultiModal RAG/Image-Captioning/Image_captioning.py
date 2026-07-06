import torch
import base64
import requests
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch

model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

processor = AutoProcessor.from_pretrained(model_name)

model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

url_image_1 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5uo16pKhdB1f2Vz7H8Utkg/image-1.png'
url_image_2 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/fsuegY1q_OxKIxNhf6zeYg/image-2.png'
url_image_3 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/KCh_pM9BVWq_ZdzIBIA9Fw/image-3.png'
url_image_4 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/VaaYLw52RaykwrE3jpFv7g/image-4.png'

image_urls = [url_image_1, url_image_2, url_image_3, url_image_4] 

def encode_images_to_base64(image_urls):
    """
    Download and encodes a list of image urls to base64 strings
    Parameters:
    image_urls: A list of image URLs.
    Returns:
    list: A list of base4-encoded image strings
    """
    encoded_images = []
    for url in image_urls:
        response = requests.get(url)
        if response.status_code==200:
            encoded_image = base64.b64encode(response.content).decode("utf-8")
            encoded_images.append(encoded_image)
            print(type(encoded_image))
        else:
            print(f"Warning: Failed to fetch image from {url} (Status code: {response.status_code})")
            encoded_images.append(None)
    return encoded_images

encoded_images = encode_images_to_base64(image_urls)

import base64
import io
from PIL import Image

def generate_model_response(
    encoded_image,
    user_query,
    assistant_prompt="You are a helpful assistant. Answer in 1 or 2 sentences."
):

    # Decode Base64 image
    image = Image.open(
        io.BytesIO(base64.b64decode(encoded_image))
    ).convert("RGB")

    prompt = assistant_prompt + "\n" + user_query

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.2,
        top_p=0.5,
        do_sample=True,
    )

    return processor.decode(
        outputs[0],
        skip_special_tokens=True
    )

user_query = "Describe the photo"

for i in range(len(encoded_images)):
    image = encoded_images[i]
    response = generate_model_response(image, user_query)
    print(response)

image = encoded_images[1]

user_query = "How many cars are in this image?"

print("User Query: ", user_query)
print("Model Response: ", generate_model_response(image, user_query))

