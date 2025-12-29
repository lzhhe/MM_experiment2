import argparse
import json
import os
import re
import shutil
import subprocess
from typing import List, Optional

import embedding
import preprocess

PAPERS_DIR = "./papers"
OUTPUT_DIR = "./outputs/pdf"
CHROMA_DIR = getattr(embedding, "CHROMA_DIR", "./outputs/chroma_db")


def add_paper_command(args):

    original_path = os.path.abspath(args.path)
    if not os.path.isfile(original_path):
        print(f"Error: PDF file not found: {original_path}")
        return

    topics_list: List[str] = []
    user_provided_topics = False
    if args.topics:
        topics_list = [t.strip() for t in args.topics.split(",") if t.strip()]
        user_provided_topics = True


    meta_path = preprocess.preprocess_pdf(original_path, OUTPUT_DIR,
                                          do_caption=True, do_detailed=True, do_ocr=True)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    text_content = meta.get("full_text_clean", "")
    if len(text_content) > 1000:
        text_content = text_content[:1000]
    paper_title = os.path.basename(original_path)

    if not args.strict:
        prompt = ""
        if user_provided_topics:
            candidate_topics = [t.strip() for t in topics_list if t.strip()]
            prompt = f"""
                You are a classifier.

                The user provides the following CANDIDATE TOPICS:
                {", ".join(candidate_topics)}

                Task:
                - Read the paper content below
                - Choose ONLY ONE topic from the candidate list
                - Reply with EXACTLY the topic text from the list
                - DO NOT add any explanation
                - DO NOT add punctuation
                - DO NOT output anything else

                If none of the topics match, reply with: Uncategorized

                Paper title: {paper_title}

                Paper content:
                {text_content}
                """
        else:
            prompt = f"""
                You are a classifier.

                Task:
                - Read the paper title and content provided below.
                - Choose up single most appropriate high-level research area that best describe the paper.
                - Output your answer like: CV
                - DO NOT add any explanation.
                - DO NOT add any punctuation except blank space between topics.
                - DO NOT output anything else.
                - DO NOT output latex word like boxed
                - Only output plain text

                If none of the topics apply, reply with: Uncategorized

                Paper title: {paper_title}

                Paper content:
                {text_content}
                """

        try:
            result = subprocess.run(
                ["ollama", "run", "qwen3:4b"],
                input=prompt,
                text=True,
                encoding="utf-8",
                errors="ignore",
                capture_output=True,
                check=True
            )

            qwen_output = result.stdout.strip()
            print("Raw Qwen output:", repr(qwen_output))

            last_line = qwen_output.splitlines()[-1].strip()
            normalized = last_line.lower()

            if user_provided_topics:

                selected = None
                for t in candidate_topics:
                    if normalized == t.lower():
                        selected = t
                        break


                if not selected:
                    tokens = re.split(r"[\s,;/]+", normalized)
                    for tok in tokens:
                        for t in candidate_topics:
                            if tok == t.lower():
                                selected = t
                                break
                        if selected:
                            break


                if not selected or normalized == "uncategorized":
                    topics_list = ["Uncategorized"]
                else:
                    topics_list = [selected]



            else:
                tokens = re.split(r"[\s,;/]+", normalized)
                tokens = [tok for tok in tokens if tok and tok.lower() != "uncategorized"]
                if not tokens:
                    topics_list = ["Uncategorized"]
                else:
                    raw_tokens = re.split(r"[\s,;/]+", last_line.strip())
                    for t in raw_tokens:
                        tt = t.strip()
                        if tt:
                            topics_list = [tt]
                            break
                    else:
                        topics_list = ["Uncategorized"]

        except subprocess.CalledProcessError as e:
            print("Warning: Qwen model classification failed.", e)
            topics_list = []

    if not topics_list:
        topics_list = ["Uncategorized"]


    main_topic = topics_list[0]
    dest_dir = os.path.join(PAPERS_DIR, main_topic)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, os.path.basename(original_path))
    shutil.move(original_path, dest_path)


    for extra_topic in topics_list[1:]:
        extra_dir = os.path.join(PAPERS_DIR, extra_topic)
        os.makedirs(extra_dir, exist_ok=True)
        extra_path = os.path.join(extra_dir, os.path.basename(dest_path))
        shutil.copy(dest_path, extra_path)

    meta["pdf_path_abs"] = os.path.abspath(dest_path)
    meta_dir = os.path.dirname(meta_path)
    meta["pdf_path_rel"] = os.path.relpath(meta["pdf_path_abs"], meta_dir).replace("\\", "/")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


    device, st_model, clip_model, clip_preprocess = embedding.init_models()
    client, col_text, col_image, col_image_text = embedding.init_chroma(CHROMA_DIR)
    embedding.upsert_text_minilm(col_text, st_model, meta)
    embedding.upsert_images_clip(col_image, device, clip_model, clip_preprocess, meta, os.path.dirname(meta_path))
    if getattr(embedding, "EMBED_IMAGE_TEXT_WITH_MINILM", False):
        embedding.upsert_image_text_minilm(col_image_text, st_model, meta)
    print(f" Added '{os.path.basename(dest_path)}' under topics: {', '.join(topics_list)}")

def bulk_add_command(args):

    input_dir = os.path.abspath(args.folder)
    if not os.path.isdir(input_dir):
        print(f"Error: not a directory: {input_dir}")
        return

    pdf_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, f))

    if not pdf_files:
        print("No PDF files found.")
        return

    print(f"Found {len(pdf_files)} PDF(s). Starting processing...")


    for i, pdf_path in enumerate(pdf_files, start=1):
        print(f"\n [{i}/{len(pdf_files)}] Processing: {pdf_path}")
        class FakeArgs:
            def __init__(self, path, topics=None, strict=False):
                self.path = path
                self.topics = topics
                self.strict = args.strict

        single_args = FakeArgs(path=pdf_path, topics=args.topics, strict=args.strict)
        try:
            add_paper_command(single_args)
        except Exception as e:
            print(f" Failed to process {pdf_path}: {e}")


def search_content(args):

    query_text = " ".join(args.query) if isinstance(args.query, list) else args.query

    device, st_model, clip_model, clip_preprocess = embedding.init_models()
    _, col_text, _, _ = embedding.init_chroma(CHROMA_DIR)

    query_vec = st_model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True)
    results = col_text.query(query_embeddings=[query_vec], n_results=args.topk,
                             include=["documents", "metadatas"])
    if not results or not results.get("documents"):
        print("No relevant papers found.")
        return
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    meta_cache: dict[str, Optional[dict]] = {}

    def get_page_from_char(meta_info: dict) -> Optional[int]:

        if meta_info.get("page", -1) != -1:
            return meta_info["page"]
        pdf_path = meta_info.get("pdf_path_abs")
        if not pdf_path:
            return None

        if pdf_path not in meta_cache:
            meta_cache[pdf_path] = None
            base_name = os.path.basename(pdf_path)
            stem = preprocess.safe_stem(base_name)

            if os.path.isdir(OUTPUT_DIR):
                for d in os.listdir(OUTPUT_DIR):
                    if d.startswith(stem + "_") and os.path.isdir(os.path.join(OUTPUT_DIR, d)):
                        meta_file = os.path.join(OUTPUT_DIR, d, "meta.json")
                        try:
                            with open(meta_file, "r", encoding="utf-8") as f:
                                data = json.load(f)
                                if data.get("pdf_path_abs") == pdf_path:
                                    meta_cache[pdf_path] = data
                                    break
                        except Exception:
                            continue
        meta_data = meta_cache.get(pdf_path)
        if not meta_data:
            return None

        char_index = meta_info.get("char_start", 0)
        pages = meta_data.get("pages", [])

        content_pages = [p["page"] for p in pages if p.get("text_clean") and p["text_clean"].strip()]
        pointer = 0
        for idx, page_num in enumerate(content_pages):
            text = pages[page_num - 1].get("text_clean", "") or ""
            if char_index < pointer + len(text):
                return page_num
            pointer += len(text)
            if idx < len(content_pages) - 1:
                # account for the "\n\n" separator between content blocks in full text
                pointer += 2
                if char_index < pointer:
                    # If char_index falls within the separator, attribute to next content page
                    return content_pages[idx + 1]
        return content_pages[-1] if content_pages else None


    for i, (doc_text, meta_info) in enumerate(zip(docs, metas), start=1):
        snippet = doc_text.replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        pdf_path = meta_info.get("pdf_path_abs", "")
        page_num = meta_info.get("page", -1)
        if page_num == -1:
            page_num = get_page_from_char(meta_info) or "?"
        display_path = pdf_path
        try:
            # Show path relative to papers directory for brevity
            display_path = os.path.relpath(pdf_path, os.path.abspath(PAPERS_DIR))
        except Exception:
            pass
        print(f"{i}. {display_path} (Page {page_num}): {snippet}")

def search_paper(args) -> list[dict]:
    """Search papers by natural language and return most relevant ones (deduplicated by paper)."""
    query_text = " ".join(args.query) if isinstance(args.query, list) else args.query

    # Initialize models and database
    device, st_model, clip_model, clip_preprocess = embedding.init_models()
    _, col_text, _, _ = embedding.init_chroma(CHROMA_DIR)


    raw_topk = max(args.topk * 5, args.topk)

    query_vec = st_model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True)

    results = col_text.query(
        query_embeddings=[query_vec],
        n_results=raw_topk,
        include=["documents", "metadatas"]
    )

    if not results or not results.get("documents"):
        print("No matching papers found.")
        return []

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    ranked_results: list[dict] = []
    seen_papers: set[str] = set()

    for doc, meta in zip(docs, metas):

        pdf_path = meta.get("pdf_path_abs") or meta.get("pdf_path") or ""
        if not pdf_path:

            pdf_path = str(meta.get("id", ""))


        if pdf_path in seen_papers:
            continue

        seen_papers.add(pdf_path)

        snippet = doc.replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."

        title = meta.get("title") or os.path.basename(pdf_path)

        print(f"{len(ranked_results) + 1}. {title}\n") # {pdf_path}\n   ⤷ {snippet}\n

        ranked_results.append({
            "rank": len(ranked_results) + 1,
            "title": title,
            # "path": pdf_path,
            # "snippet": snippet,
        })

        if len(ranked_results) >= args.topk:
            break

    if not ranked_results:
        print("No matching papers found.")
    return ranked_results


def search_image_command(args):
    query_text = " ".join(args.query) if isinstance(args.query, list) else args.query
    device, st_model, clip_model, clip_preprocess = embedding.init_models()
    _, _, _, col_image_text = embedding.init_chroma(CHROMA_DIR)
    query_vec = st_model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True)
    results = col_image_text.query(query_embeddings=[query_vec], n_results=args.topk,
                                   include=["documents", "metadatas"])
    if not results or not results.get("documents"):
        print("No matching images found.")
        return
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    for i, (desc, meta_info) in enumerate(zip(docs, metas), start=1):
        caption = (desc or "").replace("\n", " ")
        if len(caption) > 200:
            caption = caption[:200] + "..."
        pdf_path = meta_info.get("pdf_path_abs", "")
        page_no = meta_info.get("page", "?")
        image_id = meta_info.get("image_id", "")
        display_path = pdf_path
        try:
            display_path = os.path.relpath(pdf_path, os.path.abspath(PAPERS_DIR))
        except Exception:
            pass
        print(f"{i}. {display_path} (Page {page_no}, Image {image_id}): {caption}")



def list_topics_command(args):
    if not os.path.isdir(PAPERS_DIR):
        print(f"No directory found at {PAPERS_DIR}")
        return

    subdirs = [d for d in os.listdir(PAPERS_DIR)
               if os.path.isdir(os.path.join(PAPERS_DIR, d))]

    if not subdirs:
        print("No topics found.")
        return

    print("Topics and papers:\n")
    for topic in sorted(subdirs):
        topic_dir = os.path.join(PAPERS_DIR, topic)

        papers = []
        for root, _, files in os.walk(topic_dir):
            for f in files:
                if f.lower().endswith(".pdf"):
                    full_path = os.path.join(root, f)
                    # 相对 topic 目录的路径，显示更简洁
                    rel_path = os.path.relpath(full_path, topic_dir)
                    papers.append(rel_path)

        print(f"- {topic}")
        if not papers:
            print("  (no papers)")
        else:
            for i, p in enumerate(sorted(papers), start=1):
                print(f"  {i}. {p}")
        print()






def main():
    parser = argparse.ArgumentParser(
        description="Local Multimodal AI Assistant - manage and search papers/images via CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: add_paper
    add_parser = subparsers.add_parser("add_paper",
                                       help="Add a new paper PDF to the library (with optional classification).")
    add_parser.add_argument("path", help="Path to the PDF file to add.")
    add_parser.add_argument("--topics",
                            help="Optional comma-separated topics (if not provided, auto-classification will be used).")
    add_parser.add_argument("--strict", action="store_true",
                            help="If set, use all provided topics for classification instead of selecting the best one.")

    # Subcommand: bulk_add
    bulk_parser = subparsers.add_parser("bulk_add", help="Batch process and classify all PDFs in a folder.")
    bulk_parser.add_argument("folder", help="Input folder containing PDFs.")
    bulk_parser.add_argument("--topics", help="Optional candidate topics to restrict classification.")
    bulk_parser.add_argument("--strict", action="store_true",
                             help="Force assign all topics instead of selecting the best.")

    # Subcommand: search_content
    search_parser = subparsers.add_parser("search_content", help="Semantic search over paper contents.")
    search_parser.add_argument("query", nargs="+", help="Search query (natural language).")
    search_parser.add_argument("--topk", type=int, default=5, help="Number of results to return.")

    # Subcommand: search_paper
    nl_parser = subparsers.add_parser("search_paper", help="Natural language search for most relevant papers.")
    nl_parser.add_argument("query", nargs="+", help="Search query in natural language.")
    nl_parser.add_argument("--topk", type=int, default=5, help="Number of top results to return.")

    # Subcommand: search_image
    image_parser = subparsers.add_parser("search_image", help="Search images by text description.")
    image_parser.add_argument("query", nargs="+", help="Search query describing the image.")
    image_parser.add_argument("--topk", type=int, default=2, help="Number of image results to return.")

    # Subcommand: list_topics
    list_parser = subparsers.add_parser("list_topics", help="List all topic categories in the papers folder.")

    args = parser.parse_args()
    if args.command == "add_paper":
        add_paper_command(args)
    elif args.command == "bulk_add":
        bulk_add_command(args)
    elif args.command == "search_content":
        search_content(args)
    elif args.command == "search_paper":
        search_paper(args)
    elif args.command == "search_image":
        search_image_command(args)
    elif args.command == "list_topics":
        list_topics_command(args)


if __name__ == "__main__":
    main()
