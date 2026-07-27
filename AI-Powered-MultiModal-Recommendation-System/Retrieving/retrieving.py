import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

# ================================
# Verify vector database
# ================================

DB_DIR = str((Path.home() / "chroma_multimodal").resolve())

if not os.path.isdir(DB_DIR):
    raise RuntimeError(
        f"Vector database directory not found: '{DB_DIR}'. "
        "Please run Lesson 1 (Multimodal Vector Index Construction) first."
    )

article_db = Chroma(
    collection_name="restaurant_articles",
    persist_directory=DB_DIR,
)

image_db = Chroma(
    collection_name="food_images",
    persist_directory=DB_DIR,
)

n_articles = article_db._collection.count()
n_images = image_db._collection.count()

if n_articles <= 0 or n_images <= 0:
    raise RuntimeError(
        "One or more collections are empty. Please rerun Lesson 1 to rebuild the index."
    )

print(f"✅ Article vectors: {n_articles}")
print(f"✅ Image vectors:   {n_images}")


# EMBEDDING MODEL INITIALIZATION
text_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(texts, batch_size=64):
    return text_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

# image embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_name, use_fast=True)
clip_model.eval()

@torch.no_grad()
def embed_image(image_paths, batch_size=64):
    vecs = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        inputs = clip_processor(
            images=[Image.open(p).convert("RGB") for p in batch],
            return_tensors="pt"
        ).to(device)
        feats = clip_model.get_image_features(**inputs)          # (B,512)
        feats = feats / feats.norm(dim=-1, keepdim=True) 
        vecs.append(feats.cpu().numpy().astype(np.float32))
    return np.vstack(vecs)

# RETRIEVAL UTILITIES
# chroma returns
def _unwrap(res: dict):
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metadatas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return ids, docs, metadatas, dists

def print_hits(ids, docs, metas, dists, title: str, max_chars : int=180):
    for i in range(len(ids)):
        meta = metas[i] if i < len(metas) else {}
        dists = float(dists[i]) if i < len(dists) else None

        snippet = (docs[i] or "").replace("\n", " ").strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "..."

        # compact metadata view
        cuisine = meta.get("cuisine", "N/A") if isinstance(meta, dict) else "N/A"
        location = meta.get("location", "N/A") if isinstance(meta, dict) else
        doc_id = meta.get("doc_id", "N/A") if isinstance(meta, dict) else "N/A"
        source = meta.get("source", "N/A") if isinstance(meta, dict) else "N/A"

        print(f"[{i+1}] id={doc_id} | cuisine={cuisine} | location={location} | source={source} | distance={dist:.4f}")
        print(f"{snippet}")

# ARTICLE SIMILARITY RETRIEVAL
def retrieve_article(query: str, top_k: int=5, where: dict | None=None):
    query_vec = embed_text([query])
    res = article_db._collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=top_k,
        where=where,
        include=["metadatas", "documents", "distances"],
    )
    return _unwrap(res)

# IMAGE SIMILARITY RETRIEVAL
def retrieve_image(query_image_path: str, top_k: int=5, where: dict | None=None):
    query_vec = embed_image([query_image_path])
    res = image_db._collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=top_k,
        where=where,
        include=["metadatas", "documents", "distances"],
    )
    return _unwrap(res)