import os
import json
import hashlib
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import chromadb
from chromadb.config import Settings

from sentence_transformers import SentenceTransformer
import clip  # openai-clip


META_JSON_PATHS = [
    "./outputs/pdf/test1_ac4ac92f/meta.json",
]


CHROMA_DIR = "./outputs/chroma_db"


MINILM_DIR = "./models/all-MiniLM-L6-v2"
CLIP_PT_PATH = "./models/clip/ViT-B-32.pt"


TEXT_CHUNK_SIZE = 1200
TEXT_CHUNK_OVERLAP = 200

# batch
MINILM_BATCH = 64
CLIP_IMAGE_BATCH = 32


EMBED_IMAGE_TEXT_WITH_MINILM = True



def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=axis, keepdims=True) + eps
    return x / denom

def iter_text_chunks(text: str, chunk_size: int, overlap: int):
    if not text:
        return
    n = len(text)
    step = max(1, chunk_size - overlap)
    for start in range(0, n, step):
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            yield start, end, chunk
        if end >= n:
            break

def safe_join(root: str, rel: str) -> str:
    return os.path.normpath(os.path.join(root, rel))

def load_meta(meta_path: str) -> Dict[str, Any]:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)



def init_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE =", device)

    # all-MiniLM-L6-v2
    st_model = SentenceTransformer(MINILM_DIR, device=device)

    # OpenAI CLIP：直接加载你的 .pt
    # 注意：clip.load 第二个参数可以直接传 state_dict path
    clip_model, clip_preprocess = clip.load(CLIP_PT_PATH, device=device, jit=False)
    clip_model.eval()

    return device, st_model, clip_model, clip_preprocess



def init_chroma(persist_dir: str):
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )

    # 文本（MiniLM）
    col_text = client.get_or_create_collection("paper_text_minilm")

    # 图片（CLIP image embedding）
    col_image = client.get_or_create_collection("paper_image_clip")

    # 可选：图片相关文本（MiniLM，便于统一文本检索）
    col_image_text = client.get_or_create_collection("paper_image_text_minilm")

    return client, col_text, col_image, col_image_text


def upsert_text_minilm(col, st_model: SentenceTransformer, meta: Dict[str, Any]):
    pdf_abs = meta.get("pdf_path_abs", "")
    pdf_rel = meta.get("pdf_path_rel", "")
    num_pages = int(meta.get("num_pages", 0) or 0)

    ids, docs, metas = [], [], []

    # 4.1 full_text_clean -> chunks
    full_text = (meta.get("full_text_clean", "") or "").strip()
    for start, end, chunk in iter_text_chunks(full_text, TEXT_CHUNK_SIZE, TEXT_CHUNK_OVERLAP):
        uid = sha1(f"{pdf_abs}|FULL|{start}-{end}|{chunk[:64]}")
        ids.append(uid)
        docs.append(chunk)
        metas.append({
            "source": "full_text_clean_chunk",
            "pdf_path_abs": pdf_abs,
            "pdf_path_rel": pdf_rel,
            "page": -1,
            "char_start": start,
            "char_end": end,
            "num_pages": num_pages,
        })

    # 4.2 page-level text_clean
    for p in meta.get("pages", []):
        page_no = int(p.get("page", -1))
        page_text = (p.get("text_clean", "") or "").strip()
        if not page_text:
            continue
        uid = sha1(f"{pdf_abs}|PAGE|{page_no}|{page_text[:64]}")
        ids.append(uid)
        docs.append(page_text)
        metas.append({
            "source": "page_text_clean",
            "pdf_path_abs": pdf_abs,
            "pdf_path_rel": pdf_rel,
            "page": page_no,
            "num_pages": num_pages,
        })

    if not docs:
        return

    vecs = st_model.encode(
        docs,
        batch_size=MINILM_BATCH,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=vecs.tolist())



@torch.inference_mode()
def upsert_images_clip(
    col_image,
    device: str,
    clip_model,
    clip_preprocess,
    meta: Dict[str, Any],
    meta_dir: str,
):
    pdf_abs = meta.get("pdf_path_abs", "")
    pdf_rel = meta.get("pdf_path_rel", "")

    items = []
    for p in meta.get("pages", []):
        page_no = int(p.get("page", -1))
        for im in p.get("images", []):
            rel_path = im.get("file_path")
            if not rel_path:
                continue
            abs_path = safe_join(meta_dir, rel_path)
            if not os.path.exists(abs_path):
                continue

            uid = sha1(f"{pdf_abs}|IMG|{page_no}|{im.get('image_id','')}|{rel_path}")

            doc_text = " | ".join([t for t in [
                im.get("paper_caption"),
                im.get("caption"),
                im.get("detailed_caption"),
                im.get("ocr_text"),
            ] if t and str(t).strip()])[:4000]

            items.append((uid, abs_path, doc_text, {
                "source": "image",
                "pdf_path_abs": pdf_abs,
                "pdf_path_rel": pdf_rel,
                "page": page_no,
                "image_id": im.get("image_id"),
                "file_path": rel_path,
                "paper_caption": im.get("paper_caption"),
                "caption": im.get("caption"),
                "detailed_caption": im.get("detailed_caption"),
                "ocr_task": im.get("ocr_task"),
            }))

    if not items:
        return

    ids, docs, metas = [], [], []
    batch_tensors = []

    def flush():
        if not batch_tensors:
            return
        image_input = torch.cat(batch_tensors, dim=0).to(device)
        feats = clip_model.encode_image(image_input)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.float().cpu().numpy()
        feats = l2_normalize(feats)

        col_image.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=feats.tolist())

    for uid, abs_path, doc_text, md in tqdm(items, desc="CLIP image embeddings"):
        try:
            img = Image.open(abs_path).convert("RGB")
            img_tensor = clip_preprocess(img).unsqueeze(0)  # [1,3,H,W]
        except Exception:
            continue

        ids.append(uid)
        docs.append(doc_text or "")
        metas.append(md)
        batch_tensors.append(img_tensor)

        if len(batch_tensors) >= CLIP_IMAGE_BATCH:
            flush()
            ids, docs, metas, batch_tensors = [], [], [], []

    flush()


def upsert_image_text_minilm(col, st_model: SentenceTransformer, meta: Dict[str, Any]):
    pdf_abs = meta.get("pdf_path_abs", "")
    pdf_rel = meta.get("pdf_path_rel", "")

    ids, docs, metas = [], [], []

    for p in meta.get("pages", []):
        page_no = int(p.get("page", -1))
        for im in p.get("images", []):
            rel_path = im.get("file_path")
            image_id = im.get("image_id")

            candidates = [
                ("paper_caption", im.get("paper_caption")),
                ("caption", im.get("caption")),
                ("detailed_caption", im.get("detailed_caption")),
                ("ocr_text", im.get("ocr_text")),
            ]
            for kind, text in candidates:
                if not text or not str(text).strip():
                    continue
                t = str(text).strip()
                uid = sha1(f"{pdf_abs}|IMG_TXT|{page_no}|{image_id}|{kind}|{t[:64]}")

                ids.append(uid)
                docs.append(t)
                metas.append({
                    "source": "image_text",
                    "kind": kind,
                    "pdf_path_abs": pdf_abs,
                    "pdf_path_rel": pdf_rel,
                    "page": page_no,
                    "image_id": image_id,
                    "file_path": rel_path,
                })

    if not docs:
        return

    vecs = st_model.encode(
        docs,
        batch_size=MINILM_BATCH,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=vecs.tolist())


def main():
    device, st_model, clip_model, clip_preprocess = init_models()
    _, col_text, col_image, col_image_text = init_chroma(CHROMA_DIR)

    for meta_path in META_JSON_PATHS:
        meta_path = os.path.abspath(meta_path)
        meta_dir = os.path.dirname(meta_path)
        meta = load_meta(meta_path)

        print("\n=== META ===")
        print("meta_path:", meta_path)
        print("meta_dir :", meta_dir)
        print("pdf_abs  :", meta.get("pdf_path_abs", ""))

        upsert_text_minilm(col_text, st_model, meta)

        upsert_images_clip(col_image, device, clip_model, clip_preprocess, meta, meta_dir)

        if EMBED_IMAGE_TEXT_WITH_MINILM:
            upsert_image_text_minilm(col_image_text, st_model, meta)

        print("Done for:", meta_path)

    print("\n Chroma DB:", os.path.abspath(CHROMA_DIR))
    print("Collections:")
    print(" - paper_text_minilm")
    print(" - paper_image_clip")
    print(" - paper_image_text_minilm")


if __name__ == "__main__":
    main()
