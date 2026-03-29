"""
data/pope_loader.py
Module 6a: POPE Dataset Loader

POPE (Polling-based Object Probing Evaluation)
  Paper: "Evaluating Object Hallucination in Large Vision-Language Models" (Li et al., 2023)
  HuggingFace: lmms-lab/POPE

Dataset structure:
  - question_id: int
  - image:       PIL.Image (COCO images)
  - text:        str — yes/no question, e.g. "Is there a person in the image?"
  - label:       str — "yes" or "no"

Three splits:
  - adversarial: Hard negative questions using frequent co-occurring objects
  - popular:     Questions about common objects in COCO
  - random:      Random object queries

For hallucination detection:
  We generate the model's answer (yes/no) and compare with label.
  Hallucination = model says "yes" when ground truth is "no" (object not present)
"""

import logging
from typing import Iterator, Dict, Any, Optional, List
from PIL import Image

logger = logging.getLogger(__name__)

VALID_SPLITS = ["adversarial", "popular", "random"]


class POPELoader:
    """
    Loads and iterates the POPE hallucination benchmark.

    Usage:
        loader = POPELoader(config)
        for sample in loader.iter_samples(split="adversarial"):
            image = sample["image"]
            question = sample["question"]
            label = sample["label"]   # "yes" | "no"
            is_hallucination = sample["is_hallucination"]  # bool
    """

    def __init__(self, config: dict, split: Optional[str] = None):
        self.config = config
        self.data_config = config["data"]["pope"]
        self.default_split = split or self.data_config.get("default_split", "adversarial")
        self._dataset_cache = {}

    def _load_split(self, split: str):
        """Load and cache a POPE split from HuggingFace."""
        if split in self._dataset_cache:
            return self._dataset_cache[split]

        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Choose from: {VALID_SPLITS}")

        try:
            from datasets import load_dataset
            logger.info(f"Loading POPE [{split}] from HuggingFace...")
            ds = load_dataset(
                self.data_config["hf_repo"],
                split="test",
            )
            # POPE stores 'adversarial', 'popular', 'random' in the 'category' column
            if "category" in ds.column_names:
                ds = ds.filter(lambda x: x["category"] == split)
                
            self._dataset_cache[split] = ds
            logger.info(f"POPE [{split}] loaded: {len(ds)} samples.")
            return ds
        except Exception as e:
            logger.error(f"Failed to load POPE dataset: {e}")
            raise

    def iter_samples(
        self,
        split: Optional[str] = None,
        n_samples: Optional[int] = None,
        shuffle: bool = False,
    ) -> Iterator[Dict[str, Any]]:
        """
        Iterate over POPE samples.

        Yields dict with keys:
            question_id     (int)
            image           (PIL.Image)
            question        (str)
            label           (str): "yes" | "no"
            is_hallucination (bool): True if the "correct" answer is "no" (object absent)
        """
        target_split = split or self.default_split
        ds = self._load_split(target_split)

        indices = list(range(len(ds)))
        if shuffle:
            import random
            random.shuffle(indices)
        if n_samples is not None:
            indices = indices[:n_samples]

        for idx in indices:
            item = ds[idx]
            # HuggingFace POPE has "image" as PIL or path
            image = item.get("image") or item.get("img")
            if image is None:
                logger.warning(f"Sample {idx} has no image, skipping.")
                continue

            # Ensure PIL Image
            if not isinstance(image, Image.Image):
                try:
                    image = Image.open(image).convert("RGB")
                except Exception:
                    continue
            else:
                image = image.convert("RGB")

            label = item.get("label", item.get("answer", "")).strip().lower()
            question = item.get("text", item.get("question", "")).strip()

            yield {
                "question_id": item.get("question_id", idx),
                "image": image,
                "question": question,
                "label": label,
                # Object absent = "no" = model should not hallucinate presence
                "is_hallucination": (label == "no"),
            }

    def get_all_samples(
        self, split: Optional[str] = None, n_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Return all samples as a list."""
        return list(self.iter_samples(split=split, n_samples=n_samples))

    def __len__(self):
        ds = self._load_split(self.default_split)
        return len(ds)
