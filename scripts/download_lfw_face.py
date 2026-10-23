#!/usr/bin/env python3
"""Export FacePass/LFW train images and separate test queries for CLIP retrieval.

The saved catalog matches mps_clip_retrieval_v21_hybrid.py's expected layout:
    <out-dir>/<person>__ITEM_<id>/metadata.json
    <out-dir>/<person>__ITEM_<id>/images/001.jpg

Dataset columns: image (PIL), label (string), image_id (string), person_id (int).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

DATASET_NAME = "besartshyti/facepass_eval"


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return value[:90] or "person"


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/celebrity_faces"))
    parser.add_argument("--query-dir", type=Path, default=Path("queries/celebrity_faces"))
    parser.add_argument("--max-models", "--max-people", dest="max_models", type=int, default=20)
    parser.add_argument("--images-per-model", type=int, default=3)
    parser.add_argument("--skip-queries", action="store_true")
    args = parser.parse_args()

    if args.max_models <= 0 or args.images_per_model <= 0:
        parser.error("--max-models and --images-per-model must be positive")

    print(f"[INFO] Loading train split from {DATASET_NAME}...")
    train = load_dataset(DATASET_NAME, split="train", streaming=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Use the string label to link a person's train and test images.
    records: dict[str, dict] = {}
    train_image_ids: set[str] = set()

    for example in train:
        label = str(example["label"]).strip()  # e.g. George_W_Bush
        person_id = int(example["person_id"])
        image_id = str(example["image_id"])
        if not label:
            continue
        if label not in records and len(records) >= args.max_models:
            continue

        if label not in records:
            title = label.replace("_", " ")
            item_no = f"LFW_{person_id:05d}"
            model_dir = args.out_dir / f"{slugify(label)}__ITEM_{item_no}"
            (model_dir / "images").mkdir(parents=True, exist_ok=True)
            records[label] = {
                "title": title,
                "item_no": item_no,
                "label": label,
                "person_id": person_id,
                "source_dataset": DATASET_NAME,
                "source_split": "train",
                "product_url": None,
                "images": [],
                "_model_dir": model_dir,
            }

        record = records[label]
        if person_id != record["person_id"]:
            raise RuntimeError(f"Inconsistent person_id for {label}: {person_id}")
        if len(record["images"]) >= args.images_per_model:
            continue

        idx = len(record["images"]) + 1
        img_path = record["_model_dir"] / "images" / f"{idx:03d}.jpg"
        example["image"].convert("RGB").save(img_path, "JPEG", quality=92)
        record["images"].append({
            "image_url": None,
            "local_path": str(img_path),
            "source_image_id": image_id,
        })
        train_image_ids.add(image_id)
        print(f"[SAVE] {record['title']}: {idx}/{args.images_per_model}")

        if len(records) == args.max_models and all(
            len(r["images"]) == args.images_per_model for r in records.values()
        ):
            break

    manifest = []
    for label, record in sorted(records.items()):
        clean = {k: v for k, v in record.items() if not k.startswith("_")}
        clean["num_images"] = len(clean["images"])
        save_json(record["_model_dir"] / "metadata.json", clean)
        manifest.append(clean)

    save_json(args.out_dir / "_manifest.json", manifest)
    total = sum(r["num_images"] for r in manifest)
    print(f"[DONE] Saved {total} catalog images across {len(manifest)} identities")
    print(f"[DONE] Catalog: {args.out_dir}")
    if total == 0:
        raise RuntimeError("No training images obtained from the dataset")

    if not args.skip_queries:
        print("[INFO] Loading independent TEST split for holdout queries...")
        test = load_dataset(DATASET_NAME, split="test", streaming=True)
        args.query_dir.mkdir(parents=True, exist_ok=True)
        query_manifest = []
        saved_labels: set[str] = set()

        for example in test:
            label = str(example["label"]).strip()
            if label not in records or label in saved_labels:
                continue
            record = records[label]
            person_id = int(example["person_id"])
            image_id = str(example["image_id"])
            if person_id != record["person_id"]:
                raise RuntimeError(f"Train/test person_id mismatch for {label}")
            if image_id in train_image_ids:
                raise RuntimeError(f"Test image_id also used as a reference: {image_id}")

            path = args.query_dir / f"{record['item_no']}__{slugify(label)}.jpg"
            example["image"].convert("RGB").save(path, "JPEG", quality=92)
            saved_labels.add(label)
            query_manifest.append({
                "query_path": str(path),
                "expected_item_no": record["item_no"],
                "expected_title": record["title"],
                "label": label,
                "source_image_id": image_id,
            })
            print(f"[QUERY] {record['title']}")
            if len(saved_labels) == len(records):
                break

        save_json(args.query_dir / "_query_labels.json", query_manifest)
        print(f"[DONE] Saved {len(query_manifest)} holdout queries to {args.query_dir}")
        if len(saved_labels) != len(records):
            missing = sorted(set(records) - saved_labels)
            print(f"[WARN] No test image found for {len(missing)} identities: {missing}")


if __name__ == "__main__":
    main()
