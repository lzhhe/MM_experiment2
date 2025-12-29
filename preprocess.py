import hashlib
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Any
from io import BytesIO

import fitz  # PyMuPDF
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


MODEL_ID = "./models/Florence-2-base/snapshots/5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

_model = None
_processor = None


def init_florence():
    global _model, _processor
    if _model is None or _processor is None:
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True, local_files_only=True
        ).to(DEVICE)
        _model.eval()
        _processor = AutoProcessor.from_pretrained(
            MODEL_ID, trust_remote_code=True, local_files_only=True
        )



def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 断词：trans-\nformer -> transformer
    text = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", text)

    # 保留段落（双换行），其余单换行并空格
    text = text.replace("\n\n", "<PARA>")
    text = re.sub(r"\n+", " ", text)
    text = text.replace("<PARA>", "\n\n")

    text = re.sub(r"[ \t]+", " ", text).strip()
    return text



def normalize_ocr_text(text: str) -> str:
    if not text:
        return ""

    s = text.replace("\r\n", "\n").replace("\r", "\n")

    s = re.sub(r"\b([eEiI])\s*\.\s*([gGeE])\s*\.", lambda m: f"{m.group(1)}.{m.group(2)}.", s)
    s = re.sub(r"\b(i)\s*\.\s*(e)\s*\.", "i.e.", s, flags=re.IGNORECASE)

    # 断词
    s = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", s)

    # 在括号与相邻字母数字间补空格： "Agent(e.g." -> "Agent (e.g."
    s = re.sub(r"([A-Za-z0-9])\(", r"\1 (", s)
    s = re.sub(r"\)([A-Za-z0-9])", r") \1", s)

    # 标点前后空格（避免 "delay:CP"）
    s = re.sub(r"\s*:\s*", ": ", s)
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"\s*;\s*", "; ", s)

    # 多个 ! ? 的统一间隔
    s = re.sub(r"\s*([!?])\s*", r" \1 ", s)

    # 处理连在一起的词：尽量在小写->大写边界补空格（"Misaligneddelay"）
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)

    # 合并多空白
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()



def sha1_8(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


def safe_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"[^\w\-\.]+", "_", stem)
    return stem[:80]


def make_doc_output_dir(base_out_dir: str, pdf_path: str) -> str:
    return os.path.join(base_out_dir, f"{safe_stem(pdf_path)}_{sha1_8(os.path.abspath(pdf_path))}")



def load_image_for_florence(image_path: str) -> Image.Image:
    if (not os.path.exists(image_path)) or os.path.getsize(image_path) == 0:
        raise RuntimeError(f"Image not ready: {image_path}")

    with Image.open(image_path) as im:
        im = im.convert("RGB").copy()

    # 重新编码一次，确保像素确定
    buf = BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def florence_infer(image_path: str, task: str, max_new_tokens: int = 256) -> Dict[str, Any]:
    init_florence()
    img = load_image_for_florence(image_path)

    inputs = _processor(text=task, images=img, return_tensors="pt").to(DEVICE, DTYPE)
    with torch.inference_mode():
        out_ids = _model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=3,
        )

    raw = _processor.batch_decode(out_ids, skip_special_tokens=False)[0]
    parsed = _processor.post_process_generation(raw, task=task, image_size=(img.width, img.height))
    return {"task": task, "raw": raw, "parsed": parsed}


def extract_caption_from_parsed(parsed: Any, task: str) -> str:
    if isinstance(parsed, dict):
        if task in parsed and isinstance(parsed[task], str):
            return parsed[task].strip()
        for k in ("caption", "text", "description"):
            if k in parsed and isinstance(parsed[k], str):
                return parsed[k].strip()
        return json.dumps(parsed, ensure_ascii=False)
    return str(parsed).strip()


def extract_ocr_text_from_parsed(parsed: Any, task: str) -> str:
    texts: List[str] = []

    def rec(x: Any):
        if x is None:
            return
        if isinstance(x, str):
            s = x.strip()
            if s:
                texts.append(s)
            return
        if isinstance(x, (int, float, bool)):
            return
        if isinstance(x, list):
            for it in x:
                rec(it)
            return
        if isinstance(x, dict):
            if task in x and isinstance(x[task], str):
                texts.append(x[task].strip())
            for v in x.values():
                rec(v)

    rec(parsed)

    uniq, seen = [], set()
    for t in texts:
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            uniq.append(t)

    return normalize_ocr_text("\n".join(uniq))


def florence_ocr_with_fallback(image_path: str, max_new_tokens: int = 1024) -> Dict[str, Any]:
    last_err = None
    for task in ("<OCR>", "<OCR_WITH_REGION>"):
        try:
            out = florence_infer(image_path, task, max_new_tokens=max_new_tokens)
            ocr_text = extract_ocr_text_from_parsed(out["parsed"], task)
            return {"ok": True, "task_used": task, "ocr_text": ocr_text, "result": out}
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    return {"ok": False, "task_used": None, "ocr_text": "", "result": {"error": last_err or "unknown_error"}}


def sha1_16_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:16]


def render_crop_from_page(page: fitz.Page, bbox: Tuple[float, float, float, float], out_path: str, zoom: float = 3.0):
    rect = fitz.Rect(bbox)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img.save(out_path)

def bbox_iou(a, b) -> float:

    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return inter / (area_a + area_b - inter + 1e-6)


def bbox_can_merge(a, b, iou_thresh: float = 0.15, gap_thresh: float = 15.0) -> bool:

    if a is None or b is None:
        return False

    if bbox_iou(a, b) >= iou_thresh:
        return True

    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    # 水平/垂直“缝隙”
    gap_x = max(0.0, max(ax0, bx0) - min(ax1, bx1))
    gap_y = max(0.0, max(ay0, by0) - min(ay1, by1))

    return (gap_x <= gap_thresh) and (gap_y <= gap_thresh)


def cluster_image_boxes(raw_records, iou_thresh: float = 0.15, gap_thresh: float = 15.0):

    with_box = [r for r in raw_records if r["bbox"] is not None]
    no_box = [r for r in raw_records if r["bbox"] is None]

    clusters = []

    for r in with_box:
        placed = False
        for c in clusters:
            if bbox_can_merge(c["bbox"], r["bbox"], iou_thresh, gap_thresh):

                cx0, cy0, cx1, cy1 = c["bbox"]
                rx0, ry0, rx1, ry1 = r["bbox"]
                c["bbox"] = (
                    min(cx0, rx0),
                    min(cy0, ry0),
                    max(cx1, rx1),
                    max(cy1, ry1),
                )
                c["members"].append(r)
                placed = True
                break
        if not placed:
            clusters.append({"bbox": r["bbox"], "members": [r]})

    for r in no_box:
        clusters.append({"bbox": None, "members": [r]})

    return clusters

def extract_images_from_page(
    doc: fitz.Document,
    page: fitz.Page,
    page_idx: int,
    images_dir: str,
    render_zoom: float = 3.0,
):
    os.makedirs(images_dir, exist_ok=True)

    raw_records = []
    for idx, img_info in enumerate(page.get_images(full=True)):
        xref = img_info[0]

        rects = page.get_image_rects(xref)
        bbox = None
        if rects:
            r = rects[0]
            bbox = (float(r.x0), float(r.y0), float(r.x1), float(r.y1))

        raw_records.append(
            {
                "xref": xref,
                "bbox": bbox,
                "idx": idx,  # 仅用于命名
            }
        )

    clusters = cluster_image_boxes(raw_records)

    records = []

    for c_idx, cluster in enumerate(clusters, start=1):
        members = cluster["members"]
        bbox = cluster["bbox"]

        image_id = f"p{page_idx + 1}_cluster_{c_idx:03d}"

        if bbox is not None:
            abs_path = os.path.join(
                images_dir,
                f"page_{page_idx + 1:04d}_cluster_{c_idx:03d}.png",
            )
            if not os.path.exists(abs_path):
                try:
                    render_crop_from_page(page, bbox, abs_path, zoom=render_zoom)
                except Exception:
                    base = doc.extract_image(members[0]["xref"])
                    img_bytes = base["image"]
                    ext = base.get("ext", "png")
                    abs_path = os.path.join(
                        images_dir,
                        f"page_{page_idx+1:04d}_cluster_{c_idx:03d}_{sha1_16_bytes(img_bytes)}.{ext}",
                    )
                    if not os.path.exists(abs_path):
                        with open(abs_path, "wb") as f:
                            f.write(img_bytes)
        else:
            base = doc.extract_image(members[0]["xref"])
            img_bytes = base["image"]
            ext = base.get("ext", "png")
            abs_path = os.path.join(
                images_dir,
                f"page_{page_idx+1:04d}_cluster_{c_idx:03d}_{sha1_16_bytes(img_bytes)}.{ext}",
            )
            if not os.path.exists(abs_path):
                with open(abs_path, "wb") as f:
                    f.write(img_bytes)

        records.append(
            {
                "image_id": image_id,
                "page": page_idx + 1,
                "abs_path": abs_path,
                "bbox": bbox,
            }
        )

    return records


FIG_RE = re.compile(r"^\s*(Figure|Fig\.?)\s*\d+\s*[\.:]?\s*", re.IGNORECASE)


def extract_figure_captions_from_page(page: fitz.Page) -> List[Dict[str, Any]]:
    caps = []
    blocks = page.get_text("blocks")
    for b in blocks:
        x0, y0, x1, y1, text, _, block_type = b
        if block_type != 0:
            continue
        if not text:
            continue

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            continue

        head = lines[0]
        if FIG_RE.match(head):
            cap_text = " ".join(lines)
            cap_text = normalize_text(cap_text)
            caps.append({
                "bbox": (float(x0), float(y0), float(x1), float(y1)),
                "text": cap_text
            })

    return caps


def bbox_overlap_x(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax0, _, ax1, _ = a
    bx0, _, bx1, _ = b
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    union = max(ax1, bx1) - min(ax0, bx0)
    return inter / union if union > 1e-6 else 0.0


def bind_caption_to_image(
    img_bbox: Optional[Tuple[float, float, float, float]],
    captions: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:

    if img_bbox is None or not captions:
        return None

    ix0, iy0, ix1, iy1 = img_bbox


    below = []
    for cap in captions:
        cx0, cy0, cx1, cy1 = cap["bbox"]
        if cy0 >= iy1 - 2.0:
            dy = cy0 - iy1
            ox = bbox_overlap_x(img_bbox, cap["bbox"])
            below.append((dy, -ox, cap))

    if below:
        below.sort(key=lambda x: (x[0], x[1]))
        return below[0][2]


    nearest = []
    for cap in captions:
        cx0, cy0, cx1, cy1 = cap["bbox"]
        # 计算 caption 块中心
        ccx = (cx0 + cx1) / 2
        ccy = (cy0 + cy1) / 2
        icx = (ix0 + ix1) / 2
        icy = (iy0 + iy1) / 2
        dist = abs(ccy - icy) * 2.0 + abs(ccx - icx)
        ox = bbox_overlap_x(img_bbox, cap["bbox"])
        nearest.append((dist, -ox, cap))

    nearest.sort(key=lambda x: (x[0], x[1]))
    return nearest[0][2] if nearest else None



@dataclass
class ExtractedImage:
    image_id: str
    page: int
    file_path: str
    bbox: Optional[Tuple[float, float, float, float]]

    caption: Optional[str]
    detailed_caption: Optional[str]

    ocr_text: Optional[str]
    ocr_task: Optional[str]
    ocr_result: Optional[Dict[str, Any]]

    paper_caption: Optional[str]
    paper_caption_bbox: Optional[Tuple[float, float, float, float]]


def preprocess_pdf(
    pdf_path: str,
    base_out_dir: str,
    do_caption: bool = True,
    do_detailed: bool = True,
    do_ocr: bool = True,
    render_zoom: float = 3.0,
) -> str:
    output_dir = make_doc_output_dir(base_out_dir, pdf_path)
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    pages_out = []
    full_text_parts = []

    for i in range(len(doc)):
        page = doc[i]

        text_raw = page.get_text("text")
        text_clean = normalize_text(text_raw)
        full_text_parts.append(text_clean)

        figure_caps = extract_figure_captions_from_page(page)

        img_records = extract_images_from_page(doc, page, i, images_dir, render_zoom=render_zoom)
        images: List[ExtractedImage] = []

        for rec in img_records:
            rel_img_path = os.path.relpath(rec["abs_path"], output_dir).replace("\\", "/")

            bind = bind_caption_to_image(rec["bbox"], figure_caps)
            paper_caption = bind["text"] if bind else None
            paper_caption_bbox = bind["bbox"] if bind else None

            ocr_text = None
            ocr_task = None
            ocr_result = None
            if do_ocr:
                try:
                    o = florence_ocr_with_fallback(rec["abs_path"], max_new_tokens=1024)
                    ocr_text, ocr_task, ocr_result = o["ocr_text"], o["task_used"], o["result"]
                except Exception as e:
                    ocr_text, ocr_task, ocr_result = "", None, {"error": f"{type(e).__name__}: {e}"}

            cap = None
            dcap = None
            if do_caption:
                try:
                    out_cap = florence_infer(rec["abs_path"], "<CAPTION>", max_new_tokens=256)
                    cap = extract_caption_from_parsed(out_cap["parsed"], "<CAPTION>")
                except Exception as e:
                    cap = f"[CAPTION_ERROR] {type(e).__name__}: {e}"

            if do_detailed:
                try:
                    out_d = florence_infer(rec["abs_path"], "<DETAILED_CAPTION>", max_new_tokens=256)
                    dcap = extract_caption_from_parsed(out_d["parsed"], "<DETAILED_CAPTION>")
                except Exception as e:
                    dcap = f"[DETAILED_CAPTION_ERROR] {type(e).__name__}: {e}"

            images.append(ExtractedImage(
                image_id=rec["image_id"],
                page=rec["page"],
                file_path=rel_img_path,
                bbox=rec["bbox"],
                caption=cap,
                detailed_caption=dcap,
                ocr_text=ocr_text,
                ocr_task=ocr_task,
                ocr_result=ocr_result,
                paper_caption=paper_caption,
                paper_caption_bbox=paper_caption_bbox
            ))

        pages_out.append({
            "page": i + 1,
            "text_raw": text_raw,
            "text_clean": text_clean,
            "figure_captions": figure_caps,
            "images": [asdict(img) for img in images],
        })

        print(f"[PAGE {i + 1}/{len(doc)}] text={len(text_clean)} chars, images={len(images)}, fig_caps={len(figure_caps)}")

    doc.close()

    out = {
        "pdf_path_rel": os.path.relpath(pdf_path, output_dir).replace("\\", "/"),
        "pdf_path_abs": os.path.abspath(pdf_path),
        "num_pages": len(pages_out),
        "full_text_clean": "\n\n".join([t for t in full_text_parts if t.strip()]),
        "pages": pages_out,
    }

    json_path = os.path.join(output_dir, "meta.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n 完成：{json_path}")
    return json_path


if __name__ == "__main__":
    base_out_dir = "./outputs/pdf"
    pdf_list = ["./papers/test1.pdf", ]# "./papers/test2.pdf", "./papers/test3.pdf"

    for pdf_path in pdf_list:
        preprocess_pdf(
            pdf_path,
            base_out_dir,
            do_caption=True,
            do_detailed=True,
            do_ocr=True,
            render_zoom=1.0,
        )
