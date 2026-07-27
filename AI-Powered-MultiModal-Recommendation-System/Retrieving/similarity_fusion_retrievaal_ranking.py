# Standard library
import os
from pathlib import Path

# Third-party
import numpy as np
import torch
from PIL import Image
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

print("✅ Environment ready")


# ---- Text embedding model (384-d) ----
text_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts, batch_size=64):
    return text_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # cosine-ready
    ).astype(np.float32)

print("✅ Text embedder ready")


# ---- CLIP embedding model (512-d) for image + query text ----
device = "cpu"
clip_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_name, use_fast=True)
clip_model.eval()

@torch.no_grad()
def embed_images(paths, batch_size=16):
    vecs = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i+batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch]
        inputs = clip_processor(images=imgs, return_tensors="pt").to(device)
        feats = clip_model.get_image_features(**inputs)          # (B,512)
        feats = feats / feats.norm(dim=-1, keepdim=True)         # cosine-ready
        vecs.append(feats.cpu().numpy().astype(np.float32))
    return np.vstack(vecs)

@torch.no_grad()
def embed_query_clip_text(query: str):
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True).to(device)
    feats = clip_model.get_text_features(**inputs)              # (1,512)
    feats = feats / feats.norm(dim=-1, keepdim=True)            # cosine-ready
    return feats[0].cpu().numpy().astype(np.float32)

print("✅ CLIP embedders ready")


# UTILITY FUNCTION
def _unwrap(res: dict):
    """Chroma returns lists-of-lists; unwrap the first query"""

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return ids, docs, metas, dists

def _to_similarity(dists):
    """Convert 'similar is better'  distance to 'larger is better' similarity"""
    d= np.array(dists, dtype=np.float32)
    return 1.0 - d

def _min_max(x):
    """Min-max normalization to [0,1]"""
    x= np.array(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if abs(hi - lo) < 1e-8:
        return np.ones_like(x)
    return (x - lo) / (hi - lo)


def print_hits(ids, docs, metas, scores, title: str, max_chars : int=180):
    for i in range(len(ids)):
        meta = metas[i] if i < len(metas) else {}
        score = float(scores[i]) if i < len(scores) else None

        snippet = (docs[i] or "").replace("\n", " ").strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "..."

        # compact metadata view
        cuisine = meta.get("cuisine", "N/A") if isinstance(meta, dict) else "N/A"
        location = meta.get("location", "N/A") if isinstance(meta, dict) else "N/A"
        doc_id = meta.get("doc_id", "N/A") if isinstance(meta, dict) else "N/A"
        source = meta.get("source", "N/A") if isinstance(meta, dict) else "N/A"

        print(f"[{i+1}] id={doc_id} | cuisine={cuisine} | location={location} | source={source} | similarity={score:.4f}")
        print(f"{snippet}")


# RETRIEVAL FUNCTIONS
def retrieve_article(query: str, top_k: int=5, where: dict | None=None):
    query_vec = embed_texts([query])[0]
    res = article_db._collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=top_k,
        where=where,
        include=["metadatas", "documents", "distances"],
    )
    return _unwrap(res)

def retrieve_image(query_image_path: str, top_k: int=5, where: dict | None=None):
    q_vec = embed_images([query_image_path])[0]
    res = image_db._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=top_k,
        where=where,
        include=["metadatas", "documents", "distances"],
    )
    return _unwrap(res)


# IMPLEMENTING MULTIMODAL FUSION AND RERANKING
def fuse_rank(
    query: str, k_text: int =5, k_img: int =5, w_text: float=0.6, w_img:float=0.4,
    where_text: dict | None=None, where_img: dict | None=None, top_n: int=5
):
    # Retrieve top-k text and image results
    ids_text, docs_text, metas_text, dists_text = retrieve_article(query, top_k=k_text, where=where_text)
    ids_img, docs_img, metas_img, dists_img = retrieve_image(query, top_k=k_img, where=where_img)

    # Convert distances to similarities
    sims_text = _to_similarity(dists_text)
    sims_img = _to_similarity(dists_img)

    # Normalize similarities to [0,1]
    sims_text_norm = _min_max(sims_text)
    sims_img_norm = _min_max(sims_img)

    # Create a combined dictionary of results
    combined_results = []
    
    for j in range(len(ids_text)):
        combined_results.append({
            "modality": "article",
            "id": metas_text[j].get("doc_id", ids_text[j]) if isinstance(metas_text[j], dict) else ids_text[j],
            "cuisine": metas_text[j].get("cuisine", "N/A") if isinstance(metas_text[j], dict) else "N/A",
            "location": metas_text[j].get("location", "N/A") if isinstance(metas_text[j], dict) else "N/A",
            "source": metas_text[j].get("source", "N/A") if isinstance(metas_text[j], dict) else "N/A",
            "text_score": float(sims_text_norm[j]),
            "img_score": 0.0,
            "fused": float(w_text * sims_text_norm[j]),
            "snippet": (docs_text[j] or "").replace("\n", " ").strip(),
        })

    for i in range(len(ids_img)):
        if ids_img[i] in [r["id"] for r in combined_results]:
            for r in combined_results:
                if r["id"] == ids_img[i]:
                    r["img_score"] = sims_img_norm[i]
        else:
            combined_results.append({
                "modality": "image",
                "id": metas_img[i].get("doc_id", ids_img[i]) if isinstance(metas_img[i], dict) else ids_img[i],
                "cuisine": metas_img[i].get("cuisine", "N/A") if isinstance(metas_img[i], dict) else "N/A",
                "location": metas_img[i].get("location", "N/A") if isinstance(metas_img[i], dict) else "N/A",
                "source": metas_img[i].get("source", "N/A") if isinstance(metas_img[i], dict) else "N/A",
                "text_score": 0.0,
                "img_score": float(sims_img_norm[i]),
                "fused": float(w_img * sims_img_norm[i]),
                "snippet": (docs_img[i] or "").replace("\n", " ").strip(),
            })

    # Compute fused similarity scores
    combined_results.sort(key=lambda x: x["fused"], reverse=True)  # Sort by fused score descending
    if top_n is None:
        return combined_results
    top_n = max(0, min(int(top_n), len(combined_results)))
    return combined_results[:top_n]

def print_fused(rows, title: str, max_chars: int = 90):
    print(f"\n=== {title} ===")
    for idx, r in enumerate(rows, start=1):
        snippet = r["snippet"]
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "..."
        print(
            f"[{idx}] {r['modality']} | id={r['id']} | cuisine={r['cuisine']} | "
            f"location={r['location']} | fused={r['fused']:.4f} "
            f"(text={r['text_score']:.4f}, img={r['img_score']:.4f})"
        )
        print(snippet)