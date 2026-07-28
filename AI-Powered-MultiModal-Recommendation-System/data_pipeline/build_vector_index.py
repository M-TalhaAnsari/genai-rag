"""
data_pipeline/build_vector_index.py
-------------------------------------
Builds two Chroma vector collections:
  - restaurant_articles  (text embeddings via SentenceTransformer)
  - food_images          (image embeddings via CLIP)

Run once after ingest_restaurants.py and process_multimodal.py.
"""

import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DB_DIR = str((Path.home() / "chroma_multimodal").resolve())
IMAGE_DIR = "data/recipe_images"
RESTAURANT_DATA_PATH = "data/structured_restaurant_data.json"
RECIPE_DATA_PATH = "data/augmented_food_recipe.json"

# ---------------------------------------------------------------------------
# Embedding models
# ---------------------------------------------------------------------------

text_model = SentenceTransformer("all-MiniLM-L6-v2")

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
clip_model.eval()


def embed_text(texts, batch_size=64):
    return text_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype(np.float32)


@torch.no_grad()
def embed_image(image_paths, batch_size=64):
    vecs = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i : i + batch_size]
        inputs = clip_processor(
            images=[Image.open(p).convert("RGB") for p in batch],
            return_tensors="pt",
        ).to(device)
        feats = clip_model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        vecs.append(feats.cpu().numpy().astype(np.float32))
    return np.vstack(vecs)


# ---------------------------------------------------------------------------
# Document construction
# ---------------------------------------------------------------------------

def build_article_docs(restaurants: list) -> list:
    docs = []
    for i, r in enumerate(restaurants):
        name = str(r.get("name", "")).strip()
        if not name:
            continue
        text = (
            f"Restaurant: {name}\n"
            f"Cuisine: {r.get('food_style', '')}\n"
            f"Location: {r.get('location', '')}"
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "doc_id": f"rest_{i}",
                "cuisine": r.get("food_style", ""),
                "location": r.get("location", ""),
                "source": "restaurant",
            },
        ))
    return docs


def build_image_docs(image_paths: list, recipes: list) -> list:
    docs = []
    for i, (path, recipe) in enumerate(zip(image_paths, recipes)):
        docs.append(Document(
            page_content=recipe.get("name", f"recipe image {i}"),
            metadata={
                "doc_id": f"img_{i}",
                "cuisine": recipe.get("cuisine", ""),
                "source": "recipe_image",
                "recipe_id": recipe.get("id", ""),
                "image_path": path,
            },
        ))
    return docs


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------

def build_index():
    # Load data
    with open(RESTAURANT_DATA_PATH, "r") as f:
        restaurants = json.load(f)

    with open(RECIPE_DATA_PATH, "r") as f:
        recipes = json.load(f)

    image_paths = sorted(glob.glob(f"{IMAGE_DIR}/**/*.png", recursive=True))
    print(f"Restaurants: {len(restaurants)} | Recipes: {len(recipes)} | Images: {len(image_paths)}")

    article_docs = build_article_docs(restaurants)
    image_docs = build_image_docs(image_paths, recipes)

    # Rebuild DB from scratch
    if os.path.isdir(DB_DIR):
        shutil.rmtree(DB_DIR)

    # Article collection
    A = embed_text([d.page_content for d in article_docs])
    article_db = Chroma(collection_name="restaurant_articles", persist_directory=DB_DIR)
    article_db._collection.upsert(
        ids=[d.metadata["doc_id"] for d in article_docs],
        embeddings=A.tolist(),
        documents=[d.page_content for d in article_docs],
        metadatas=[d.metadata for d in article_docs],
    )
    print(f"Article vectors stored: {len(article_docs)}")

    # Image collection
    V = embed_image([d.metadata["image_path"] for d in image_docs])
    image_db = Chroma(collection_name="food_images", persist_directory=DB_DIR)
    image_db._collection.upsert(
        ids=[d.metadata["doc_id"] for d in image_docs],
        embeddings=V.tolist(),
        documents=[d.page_content for d in image_docs],
        metadatas=[d.metadata for d in image_docs],
    )
    print(f"Image vectors stored: {len(image_docs)}")
    print(f"Index saved → {DB_DIR}")


if __name__ == "__main__":
    build_index()
