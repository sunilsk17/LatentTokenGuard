"""
data/coco_indoor_loader.py
Module 6c: COCO Indoor Subset Loader — Novel Dataset Contribution ⭐

This is one of the paper's original contributions.

Design:
  1. Load COCO val2017 annotations (instances_val2017.json)
  2. Filter images containing indoor categories (chair, bed, TV, etc.)
  3. For each image, generate:
     a. POSITIVE samples: "Is there a {object}?" → label="yes" (faithful)
     b. NEGATIVE samples: "Is there a {object}?" where object is NOT in image → label="no" (hallucination candidate)
  4. Pairs are balanced (50/50 by default)

Result: A POPE-style binary hallucination benchmark tailored to indoor scenes,
        allowing evaluation of room-understanding capabilities.

Usage:
    loader = COCOIndoorLoader(config)
    for sample in loader.iter_samples():
        ...
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, List, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

# Indoor COCO category names (subset of 80 COCO classes)
INDOOR_CATEGORIES = [
    "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors"
]


class COCOIndoorLoader:
    """
    Custom indoor COCO hallucination benchmark.

    Generates balanced yes/no questions about indoor objects
    from COCO val2017 images and annotations.
    """

    def __init__(self, config: dict, seed: int = 42):
        self.config = config
        self.data_config = config["data"]["coco"]
        self.images_dir = Path(self.data_config["images_dir"])
        self.ann_file = Path(self.data_config["annotations_file"])
        self.indoor_cats = self.data_config.get("indoor_categories", INDOOR_CATEGORIES)
        self.samples_per_cat = self.data_config.get("samples_per_category", 20)
        self.positive_ratio = self.data_config.get("positive_ratio", 0.5)
        self.seed = seed
        self._samples = None

    def _check_paths(self):
        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"COCO images not found at: {self.images_dir}\n"
                f"  → Extract val2017.zip into: {self.images_dir.parent}"
            )
        if not self.ann_file.exists():
            raise FileNotFoundError(
                f"COCO annotations not found at: {self.ann_file}\n"
                f"  → Run: python data/download_coco.py"
            )

    def _load_annotations(self) -> Tuple[dict, dict, dict]:
        """
        Load COCO annotations and build lookup structures.

        Returns:
            cat_name_to_id:  {category_name: category_id}
            img_to_cats:     {image_id: set of category_ids present in image}
            img_id_to_info:  {image_id: {"file_name": ..., "width": ..., "height": ...}}
        """
        self._check_paths()
        logger.info(f"Loading COCO annotations from {self.ann_file} ...")

        with open(self.ann_file, "r") as f:
            coco_data = json.load(f)

        # Category name → ID mapping (only indoor)
        cat_name_to_id = {
            cat["name"]: cat["id"]
            for cat in coco_data["categories"]
            if cat["name"] in self.indoor_cats
        }
        cat_id_to_name = {v: k for k, v in cat_name_to_id.items()}
        indoor_cat_ids = set(cat_name_to_id.values())

        # Image ID → set of indoor category IDs
        img_to_cats: Dict[int, set] = {}
        for ann in coco_data["annotations"]:
            if ann["category_id"] in indoor_cat_ids:
                img_to_cats.setdefault(ann["image_id"], set()).add(ann["category_id"])

        # Image ID → file info
        img_id_to_info = {
            img["id"]: img
            for img in coco_data["images"]
            if img["id"] in img_to_cats
        }

        logger.info(
            f"Found {len(img_to_cats)} indoor images across "
            f"{len(cat_name_to_id)} indoor categories."
        )
        return cat_name_to_id, cat_id_to_name, img_to_cats, img_id_to_info

    def _build_samples(self) -> List[Dict[str, Any]]:
        """Build the full list of yes/no hallucination samples."""
        if self._samples is not None:
            return self._samples

        cat_name_to_id, cat_id_to_name, img_to_cats, img_id_to_info = self._load_annotations()
        rng = random.Random(self.seed)

        # Organize images by category
        cat_id_to_images: Dict[int, List[int]] = {}
        for img_id, cat_ids in img_to_cats.items():
            for cat_id in cat_ids:
                cat_id_to_images.setdefault(cat_id, []).append(img_id)

        all_img_ids = list(img_id_to_info.keys())
        samples = []
        question_id = 0

        for cat_name in self.indoor_cats:
            cat_id = cat_name_to_id.get(cat_name)
            if cat_id is None:
                continue  # Not found in this COCO split

            images_with_cat = cat_id_to_images.get(cat_id, [])
            images_without_cat = [
                img_id for img_id in all_img_ids
                if cat_id not in img_to_cats.get(img_id, set())
            ]

            n_per_cat = self.samples_per_cat
            n_positive = int(n_per_cat * self.positive_ratio)
            n_negative = n_per_cat - n_positive

            # ── POSITIVE samples: object IS in the image ──────────────────────
            pos_imgs = rng.sample(images_with_cat, min(n_positive, len(images_with_cat)))
            for img_id in pos_imgs:
                info = img_id_to_info[img_id]
                samples.append({
                    "question_id": question_id,
                    "image_id": img_id,
                    "image_path": str(self.images_dir / info["file_name"]),
                    "question": f"Is there a {cat_name} in the image?",
                    "label": "yes",
                    "category": cat_name,
                    "is_hallucination": False,
                })
                question_id += 1

            # ── NEGATIVE samples: object is NOT in the image ──────────────────
            neg_imgs = rng.sample(images_without_cat, min(n_negative, len(images_without_cat)))
            for img_id in neg_imgs:
                info = img_id_to_info[img_id]
                samples.append({
                    "question_id": question_id,
                    "image_id": img_id,
                    "image_path": str(self.images_dir / info["file_name"]),
                    "question": f"Is there a {cat_name} in the image?",
                    "label": "no",
                    "category": cat_name,
                    "is_hallucination": True,   # "no" = model should NOT say object exists
                })
                question_id += 1

        rng.shuffle(samples)
        self._samples = samples
        logger.info(f"COCO Indoor: built {len(samples)} total samples.")
        return samples

    def iter_samples(
        self,
        n_samples: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Iterate over COCO Indoor samples.

        Yields dict with keys:
            question_id     (int)
            image           (PIL.Image)
            question        (str): e.g. "Is there a chair in the image?"
            label           (str): "yes" | "no"
            category        (str): Indoor category name
            is_hallucination (bool)
        """
        samples = self._build_samples()
        if n_samples is not None:
            samples = samples[:n_samples]

        for s in samples:
            try:
                image = Image.open(s["image_path"]).convert("RGB")
            except Exception as e:
                logger.warning(f"Could not load image {s['image_path']}: {e}")
                continue

            yield {**s, "image": image}

    def get_all_samples(self, n_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        return list(self.iter_samples(n_samples=n_samples))

    def get_stats(self) -> Dict[str, Any]:
        """Return dataset statistics."""
        samples = self._build_samples()
        n_yes = sum(1 for s in samples if s["label"] == "yes")
        n_no = len(samples) - n_yes
        cats = list({s["category"] for s in samples})
        return {
            "total": len(samples),
            "positive (yes)": n_yes,
            "negative (no)": n_no,
            "categories": sorted(cats),
        }

    def __len__(self):
        return len(self._build_samples())
