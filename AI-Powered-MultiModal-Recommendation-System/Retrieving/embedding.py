
# Required library
import glob
import json
import os
import shutil
from pathlib import Path

# Third-party library
import numpy as np
import torch
from PIL import Image
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

# prepare the image dataset
# ================================
# Download and prepare image data
# ================================
ZIP_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5_Rr6ohviItzucyWk6nkrw/synthetic-recipe-images.zip"
ZIP_PATH = "synthetic-recipe-images.zip"
IMG_DIR  = "recipe_images"

image_paths = sorted(glob.glob(f"{IMG_DIR}/**/*.png", recursive=True))
print(f"✅ Images found: {len(image_paths)}")


# Load the structureda data
with open("AI-Powered-MultiModal-Recommendation-System\data\structured_restaurant_data.json", "r") as f:
    restaurants = json.load(f)

with open("AI-Powered-MultiModal-Recommendation-System\data\augmented_food_recipe.json", "r") as f:
    recipes = json.load(f)


# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text, batch_size=64):
    """Embed the text using the embedding model."""
    return embedding_model.encode(text, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True).astype(np.float32)
device = "cpu"
clip_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_name, use_fast=True)
clip_model.eval()

@torch.no_grad()
def embed_image(image_path, batch_size=64):
    vecs =[]
    for i in range(0, len(image_path), batch_size):
        batch = image_path[i:i + batch_size]
        inputs = clip_processor(images=[Image.open(p).convert("RGB") for p in batch], return_tensors="pt").to(device)
        feats = clip_model.get_image_features(**inputs)          # (B,512)
        feats = feats / feats.norm(dim=-1, keepdim=True) 
        vecs.append(feats.cpu().numpy().astype(np.float32))
    return np.vstack(vecs)

# Construct Multimodal document
"""
Constructing Two document set 
1. Article Documents
2. Image Documents
"""

# Article
article_docs = []
for i,r in enumerate(restaurants):
    name = str(r.get("name", "")).strip()
    if not name:
        continue
    text = (
        f"Restaurant: {name}\n",
        f"Cuisine: {r.get('food_style', '')}\n",
        f"Location: {r.get('location', '')}\n"
    )

    doc_id = f"rest_{i}"

    article_docs.append(Document(
        page_content=text.strip(),
        metadata={"doc_id": doc_id, 
                  "cuisine": r.get("food_style", ""),
                  "location":r.get("location", ""),
                  "source": "restaurant"},
    ))

# Images
images_doc = []
for i, (p,rec) in enumerate(zip(image_paths, recipes)):
    doc_id = f"img_{i}"
    images_doc.append(Document(
        page_content=rec.get("name", f"recip image {i}"),
        metadata={"doc_id": doc_id, 
                  "cuisine": rec.get("cuisine", ""),
                  "source": "recipe_image",
                  "recipe_id": rec.get("id", ""),
                  "image_path": p},
    ))


# Construct The Vector Index
DB_DIR = str((Path.home() / "chroma_multimodal").resolve())

if os.path.isdir(DB_DIR):
    shutil.rmtree(DB_DIR)

A = embed_text([d.page_content for d in article_docs], batch_size=64)

artcile_db = Chroma(
    collection_name="restaurant_articles",
    persistent_directory=DB_DIR,
)

artcile_db._collection.upsert(
    ids=[d.metadata["doc_id"] for d in article_docs],
    embeddings=A.tolist(),
    documents=[d.page_content for d in article_docs],
    metadatas=[d.metadata for d in article_docs]
)

V = embed_image([d.metadata["image_path"] for d in images_doc], batch_size=64)
image_db = Chroma(
    collection_name="food_images",
    persistent_directory=DB_DIR,
)
image_db._collection.upsert(
    ids=[d.metadata["doc_id"] for d in images_doc],
    embeddings=V.tolist(),
    documents=[d.page_content for d in images_doc],
    metadatas=[d.metadata for d in images_doc]
)