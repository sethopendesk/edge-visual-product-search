#!/usr/bin/env python3
"""Download a small FGVC-Aircraft subset in the existing CLIP catalog layout.

Example (from the project root on Windows):
    python scripts/download_fgvc_aircraft.py --out-dir data/fgvc_aircraft \
        --query-dir queries/fgvc_aircraft --max-models 20 --images-per-model 3

The saved FGVC IDs are synthetic class identifiers, NOT Tamiya item numbers.
FGVC-Aircraft photographs are limited to non-commercial research by the
original dataset creators. Review their terms before using in other contexts.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

DATASET_NAME = "Donghyun99/FGVC-Aircraft"


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return value[:90] or "aircraft"


def class_name(ds, field: str, label_id: int) -> str:
    feature = ds.features[field]
    return feature.int2str(int(label_id))


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/fgvc_aircraft"))
    parser.add_argument("--query-dir", type=Path, default=Path("queries/fgvc_aircraft"))
    parser.add_argument("--max-models", type=int, default=20)
    parser.add_argument("--images-per-model", type=int, default=3)
    parser.add_argument("--skip-queries", action="store_true")
    args = parser.parse_args()

    if args.max_models <= 0 or args.images_per_model <= 0:
        parser.error("--max-models and --images-per-model must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Streaming train split from {DATASET_NAME}...")
    train = load_dataset(DATASET_NAME, split="train", streaming=True)

    # Keyed by the label ID. Avoid confusing these IDs with Tamiya item numbers.
    records: dict[int, dict] = {}

    for example in train:
        variant_id = int(example["variation"])
        if variant_id not in records and len(records) >= args.max_models:
            continue

        if variant_id not in records:
            variant = class_name(train, "variation", variant_id)
            manufacturer = class_name(train, "manufacturer", int(example["manufacturer"]))
            title = (
                variant if variant.lower().startswith(manufacturer.lower())
                else f"{manufacturer} {variant}"
            )
            item_no = f"FGVC_{variant_id:03d}"
            model_dir = args.out_dir / f"{slugify(title)}__ITEM_{item_no}"
            (model_dir / "images").mkdir(parents=True, exist_ok=True)
            records[variant_id] = {
                "title": title,
                "item_no": item_no,
                "manufacturer": manufacturer,
                "variant": variant,
                "variant_id": variant_id,
                "source_dataset": DATASET_NAME,
                "source_split": "train",
                "product_url": None,
                "images": [],
                "_model_dir": model_dir,
            }

        record = records[variant_id]
        if len(record["images"]) >= args.images_per_model:
            continue

        idx = len(record["images"]) + 1
        img_path = record["_model_dir"] / "images" / f"{idx:03d}.jpg"
        example["image"].convert("RGB").save(img_path, "JPEG", quality=92)
        record["images"].append({"image_url": None, "local_path": str(img_path)})
        print(f"[SAVE] {record['title']}: {idx}/{args.images_per_model}")

        if len(records) == args.max_models and all(
            len(r["images"]) == args.images_per_model for r in records.values()
        ):
            break

    manifest = []
    for variant_id, record in sorted(records.items()):
        clean = {k: v for k, v in record.items() if not k.startswith("_")}
        clean["num_images"] = len(clean["images"])
        save_json(record["_model_dir"] / "metadata.json", clean)
        manifest.append(clean)

    save_json(args.out_dir / "_manifest.json", manifest)
    total = sum(r["num_images"] for r in manifest)
    print(f"[DONE] Saved {total} catalog images across {len(manifest)} variants")
    print(f"[DONE] Catalog: {args.out_dir}")

    if not args.skip_queries and records:
        args.query_dir.mkdir(parents=True, exist_ok=True)
        print("[INFO] Streaming a separate TEST split for holdout query images...")
        test = load_dataset(DATASET_NAME, split="test", streaming=True)
        query_manifest = []
        saved_ids = set()
        for example in test:
            variant_id = int(example["variation"])
            if variant_id not in records or variant_id in saved_ids:
                continue
            record = records[variant_id]
            path = args.query_dir / f"{record['item_no']}__{slugify(record['title'])}.jpg"
            example["image"].convert("RGB").save(path, "JPEG", quality=92)
            saved_ids.add(variant_id)
            query_manifest.append({
                "query_path": str(path),
                "expected_item_no": record["item_no"],
                "expected_title": record["title"],
            })
            print(f"[QUERY] {record['title']}")
            if len(saved_ids) == len(records):
                break
        save_json(args.query_dir / "_query_labels.json", query_manifest)
        print(f"[DONE] Saved {len(query_manifest)} holdout queries to {args.query_dir}")

    if total == 0:
        raise RuntimeError("No images obtained from the dataset")


if __name__ == "__main__":
    main()
