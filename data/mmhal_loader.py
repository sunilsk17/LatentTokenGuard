"""
data/mmhal_loader.py
Module 6b: MMHal-Bench Dataset Loader

MMHal-Bench (Multimodal Hallucination Benchmark)
  Paper: "Aligning Large Multimodal Models with Factually Augmented RLHF" (Sun et al., 2023)
  HuggingFace: Shengcao1006/MMHal-Bench

Dataset structure:
  - 96 questions based on OpenImages
  - 12 object topics × 8 question types
  - 8 hallucination error categories
  - Requires GPT-4 for full scoring (we provide binary approximation)

Binary approximation for paper:
  We treat samples with hallucination_type != 'none' as hallucinated.
  This avoids GPT-4 dependency while still enabling F1/AUROC comparison.
"""

import logging
from typing import Iterator, Dict, Any, Optional, List
from PIL import Image

logger = logging.getLogger(__name__)


class MMHalLoader:
    """
    Loads and iterates the MMHal-Bench dataset.

    Usage:
        loader = MMHalLoader(config)
        for sample in loader.iter_samples():
            image = sample["image"]
            question = sample["question"]
            answer = sample["gt_answer"]
            is_hallucination = sample["is_hallucination"]
    """

    def __init__(self, config: dict):
        self.config = config
        self.data_config = config["data"]["mmhal"]
        self._dataset = None

    def _load(self):
        """Load MMHal-Bench from HuggingFace."""
        if self._dataset is not None:
            return self._dataset

        try:
            from datasets import load_dataset
            logger.info("Loading MMHal-Bench from HuggingFace...")
            ds = load_dataset(
                self.data_config["hf_repo"],
            )
            # MMHal-Bench is a single split (no train/test)
            if isinstance(ds, dict):
                split_key = list(ds.keys())[0]
                self._dataset = ds[split_key]
            else:
                self._dataset = ds
            logger.info(f"MMHal-Bench loaded: {len(self._dataset)} samples.")
            return self._dataset
        except Exception as e:
            logger.error(f"Failed to load MMHal-Bench: {e}")
            raise

    def iter_samples(
        self,
        n_samples: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Iterate over MMHal-Bench samples.

        Yields dict with keys:
            sample_id        (int)
            image            (PIL.Image)
            question         (str)
            gt_answer        (str): Human-annotated ground truth answer
            gt_image_content (str): Description of what's in the image
            question_type    (str): One of 8 question types
            is_hallucination (bool): True if hallucination_type != "none" (binary approx)
        """
        ds = self._load()

        cnt = 0
        for idx, item in enumerate(ds):
            if n_samples is not None and cnt >= n_samples:
                break

            # Load image
            image = item.get("image") or item.get("img")
            if image is None:
                # Try to load from URL if available
                img_url = item.get("image_url", "")
                if img_url:
                    try:
                        import requests
                        from io import BytesIO
                        resp = requests.get(img_url, timeout=10)
                        image = Image.open(BytesIO(resp.content)).convert("RGB")
                    except Exception:
                        logger.warning(f"Could not load image for sample {idx}, skipping.")
                        continue
                else:
                    logger.warning(f"Sample {idx} has no image, skipping.")
                    continue

            if isinstance(image, Image.Image):
                image = image.convert("RGB")

            # Hallucination detection label
            # MMHal has a 'hallucination' or 'response_score' field depending on version
            hall_type = item.get("hallucination", item.get("hallucination_type", "")).strip()
            # Treat non-empty hallucination type as hallucinated
            is_hallucination = bool(hall_type and hall_type.lower() not in ("", "none", "nan"))

            yield {
                "sample_id": idx,
                "image": image,
                "question": item.get("question", "").strip(),
                "gt_answer": item.get("gt_answer", item.get("answer", "")).strip(),
                "gt_image_content": item.get("gt_image_content", "").strip(),
                "question_type": item.get("question_type", "unknown"),
                "hallucination_type": hall_type,
                "is_hallucination": is_hallucination,
            }
            cnt += 1

    def get_all_samples(self, n_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return all samples as a list."""
        return list(self.iter_samples(n_samples=n_samples))

    def __len__(self):
        ds = self._load()
        return len(ds)
