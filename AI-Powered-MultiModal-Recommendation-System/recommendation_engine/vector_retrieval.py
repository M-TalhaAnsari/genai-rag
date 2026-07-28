"""
recommendation_engine/vector_retrieval.py
-------------------------------------------
Retrieval utilities for both modalities:
  - retrieve_by_text   — query restaurant articles with a text string
  - retrieve_by_image  — query recipe images with an image file
  - fuse_and_rank      — multimodal fusion + reranking

Loaded once and reused by the agent workflow.
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
# Paths & DB connection
# ---------------------------------------------------------------------------

DB_DIR = str((Path.home() / "chroma_multimodal").resolve())

if not os.path.isdir(DB_DIR):
    raise RuntimeError(
        f"Vector DB not found at '{DB_DIR}'. "
        "Run data_pipeline/build_vector_index.py first."
    )

article_db = Chroma(collection_name="restaurant_articles", persist_directory=DB_DIR)
image_db = Chroma(collection_name="food_images", persist_directory=DB_DIR)

n_articles = article_db._collection.count()
n_images = image_db._collection.count()

if n_articles <= 0 or n_images <= 0:
    raise RuntimeError("One or more Chroma collections are empty. Rebuild the index.")

print(f"Vector DB loaded — articles: {n_articles}, images: {n_images}")

# ---------------------------------------------------------------------------
# Embedding models
# ---------------------------------------------------------------------------

text_model = SentenceTransformer("all-MiniLM-L6-v2")

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
clip_model.eval()


def _embed_texts(texts, batch_size=64):
    return text_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype(np.float32)


@torch.no_grad()
def _embed_images(paths, batch_size=64):
    vecs = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i : i + batch_size]
        inputs = clip_processor(
            images=[Image.open(p).convert("RGB") for p in batch],
            return_tensors="pt",
        ).to(device)
        feats = clip_model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        vecs.append(feats.cpu().numpy().astype(np.float32))
    return np.vstack(vecs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unwrap(res: dict):
    """Chroma returns lists-of-lists; unwrap the first query."""
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return ids, docs, metas, dists


def _to_similarity(dists):
    """Convert distance (smaller = better) to similarity (larger = better)."""
    return 1.0 - np.array(dists, dtype=np.float32)


def _min_max(x):
    x = np.array(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if abs(hi - lo) < 1e-8:
        return np.ones_like(x)
    return (x - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Public retrieval functions
# ---------------------------------------------------------------------------

def retrieve_by_text(query: str, top_k: int = 5, where: dict = None):
    """Retrieve restaurant articles by text query. Returns (ids, docs, metas, dists)."""
    query_vec = _embed_texts([query])[0]
    res = article_db._collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=top_k,
        where=where,
        include=["metadatas", "documents", "distances"],
    )
    return _unwrap(res)


def retrieve_by_image(image_path: str, top_k: int = 5, where: dict = None):
    """Retrieve recipe images by visual similarity. Returns (ids, docs, metas, dists)."""
    query_vec = _embed_images([image_path])[0]
    res = image_db._collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=top_k,
        where=where,
        include=["metadatas", "documents", "distances"],
    )
    return _unwrap(res)


def fuse_and_rank(
    query: str,
    k_text: int = 5,
    k_img: int = 5,
    w_text: float = 0.6,
    w_img: float = 0.4,
    where_text: dict = None,
    where_img: dict = None,
    top_n: int = 5,
) -> list:
    """
    Retrieve from both modalities, fuse scores, and return ranked results.

    Returns a list of dicts:
      id, modality, cuisine, location, source, text_score, img_score, fused, snippet
    """
    ids_text, docs_text, metas_text, dists_text = retrieve_by_text(query, k_text, where_text)
    ids_img, docs_img, metas_img, dists_img = retrieve_by_image(query, k_img, where_img)

    sims_text = _min_max(_to_similarity(dists_text))
    sims_img = _min_max(_to_similarity(dists_img))

    combined: list[dict] = []

    for j, doc_id in enumerate(ids_text):
        meta = metas_text[j] if isinstance(metas_text[j], dict) else {}
        combined.append({
            "modality": "article",
            "id": meta.get("doc_id", doc_id),
            "cuisine": meta.get("cuisine", "N/A"),
            "location": meta.get("location", "N/A"),
            "source": meta.get("source", "N/A"),
            "text_score": float(sims_text[j]),
            "img_score": 0.0,
            "fused": float(w_text * sims_text[j]),
            "snippet": (docs_text[j] or "").replace("\n", " ").strip(),
        })

    existing_ids = {r["id"] for r in combined}

    for i, doc_id in enumerate(ids_img):
        meta = metas_img[i] if isinstance(metas_img[i], dict) else {}
        rid = meta.get("doc_id", doc_id)

        if rid in existing_ids:
            for r in combined:
                if r["id"] == rid:
                    r["img_score"] = float(sims_img[i])
                    r["fused"] += float(w_img * sims_img[i])
        else:
            combined.append({
                "modality": "image",
                "id": rid,
                "cuisine": meta.get("cuisine", "N/A"),
                "location": meta.get("location", "N/A"),
                "source": meta.get("source", "N/A"),
                "text_score": 0.0,
                "img_score": float(sims_img[i]),
                "fused": float(w_img * sims_img[i]),
                "snippet": (docs_img[i] or "").replace("\n", " ").strip(),
            })

    combined.sort(key=lambda x: x["fused"], reverse=True)
    top_n = max(0, min(int(top_n), len(combined)))
    return combined[:top_n]


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------

def print_results(rows: list, title: str = "Results", max_chars: int = 120):
    print(f"\n=== {title} ===")
    for idx, r in enumerate(rows, start=1):
        snippet = r["snippet"]
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "…"
        score_info = (
            f"fused={r['fused']:.4f} (text={r['text_score']:.4f}, img={r['img_score']:.4f})"
            if "fused" in r
            else f"distance={r.get('distance', '?')}"
        )
        print(
            f"[{idx}] {r.get('modality', '')} | id={r['id']} | "
            f"cuisine={r['cuisine']} | location={r['location']} | {score_info}"
        )
        print(f"    {snippet}")
