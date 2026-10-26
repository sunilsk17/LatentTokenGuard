"""
caption_bench/chair.py
CHAIR-style object extraction and labelling for free-form captions.

Follows the CHAIR metric of Rohrbach et al. (EMNLP 2018):
  - the 80 MSCOCO classes plus their synonym list
    (resources/chair_synonyms.txt, from github.com/LisaAnne/Hallucination)
  - the same hard-coded phrase rules ("baby elephant" is an elephant, not a
    person; "passenger jet" is an airplane; "toilet seat" is not a chair)
  - ground-truth objects = instance annotations plus objects named in the
    reference captions

Unlike the original script, every mention keeps its character span so it
can be mapped back to the generated token that introduced it.
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set

SYNONYMS_FILE = Path(__file__).parent / "resources" / "chair_synonyms.txt"

# Phrase rules from the CHAIR reference implementation
ANIMAL_WORDS = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
                "bear", "zebra", "giraffe", "animal", "cub"]
VEHICLE_WORDS = ["jet", "train"]

# Plurals that simple suffix rules get wrong
IRREGULAR_PLURALS = {
    "people": "person", "men": "man", "women": "woman", "children": "child",
    "mice": "mouse", "knives": "knife", "geese": "goose", "oxen": "ox",
    "teeth": "tooth", "feet": "foot", "calves": "calf", "wolves": "wolf",
    "shelves": "shelf", "leaves": "leaf", "loaves": "loaf", "halves": "half",
    "scarves": "scarf", "buses": "bus", "glasses": "glass", "benches": "bench",
    "sandwiches": "sandwich", "couches": "couch", "boxes": "box",
    "dishes": "dish", "brushes": "brush", "toothbrushes": "toothbrush",
    "peaches": "peach", "radishes": "radish", "foxes": "fox",
    "ponies": "pony", "puppies": "puppy", "skis": "skis", "scissors": "scissors",
}

_WORD_RE = re.compile(r"[a-z]+")


def singularize(word: str) -> str:
    """Rule-based singular form, good enough for the COCO object vocabulary."""
    if word in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[word]
    if len(word) <= 3 or word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "xes", "zes", "sses")):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


@dataclass
class Mention:
    """One object mention in a caption."""
    surface: str      # text as written, e.g. "teddy bears"
    canonical: str    # COCO class, e.g. "teddy bear"
    start: int        # character span in the caption
    end: int
    label: int = -1   # 1 = hallucinated, 0 = real, -1 = not labelled yet
    first: bool = True  # first mention of this class in the caption

    def to_dict(self) -> dict:
        return asdict(self)


class ChairExtractor:
    """
    Finds COCO objects in text.

    Usage:
        chair = ChairExtractor()
        mentions = chair.extract("Two dogs sit next to a teddy bear.")
        # → [Mention(surface='dogs', canonical='dog', ...),
        #    Mention(surface='teddy bear', canonical='teddy bear', ...)]
    """

    def __init__(self, synonyms_file: Path = SYNONYMS_FILE):
        self.phrase_to_class: Dict[str, Optional[str]] = {}
        self.classes: List[str] = []
        self._load_synonyms(synonyms_file)
        self._add_phrase_rules()
        self.max_phrase_len = max(len(p.split()) for p in self.phrase_to_class)

    def _load_synonyms(self, path: Path):
        for line in Path(path).read_text().splitlines():
            items = [s.strip().lower() for s in line.split(",") if s.strip()]
            if not items:
                continue
            canonical = items[0]
            self.classes.append(canonical)
            for item in items:
                self.phrase_to_class[item] = canonical

    def _add_phrase_rules(self):
        for animal in ANIMAL_WORDS:
            target = self.phrase_to_class.get(animal)   # "animal"/"cub" → None
            self.phrase_to_class[f"baby {animal}"] = target
            self.phrase_to_class[f"adult {animal}"] = target
        for vehicle in VEHICLE_WORDS:
            self.phrase_to_class[f"passenger {vehicle}"] = self.phrase_to_class[vehicle]
        self.phrase_to_class["bow tie"] = "tie"
        self.phrase_to_class["toilet seat"] = "toilet"

    def _lookup(self, words: List[str]) -> Optional[str]:
        """Class for a phrase, trying the last word as written and singularized."""
        last_forms = {words[-1], singularize(words[-1])}
        for last in last_forms:
            phrase = " ".join(words[:-1] + [last])
            if phrase in self.phrase_to_class:
                return phrase
        return None

    def extract(self, text: str) -> List[Mention]:
        """Return object mentions in reading order, longest phrase first."""
        lowered = text.lower()
        words = [(m.group(), m.start(), m.end()) for m in _WORD_RE.finditer(lowered)]

        mentions: List[Mention] = []
        i = 0
        while i < len(words):
            matched = False
            for n in range(min(self.max_phrase_len, len(words) - i), 0, -1):
                window = words[i:i + n]
                phrase = self._lookup([w for w, _, _ in window])
                if phrase is None:
                    continue
                canonical = self.phrase_to_class[phrase]
                if canonical is not None:
                    start, end = window[0][1], window[-1][2]
                    mentions.append(Mention(text[start:end], canonical, start, end))
                i += n
                matched = True
                break
            if not matched:
                i += 1

        # CHAIR rule: "seat" next to a toilet is the toilet seat, not a chair
        if any(m.canonical == "toilet" for m in mentions):
            mentions = [m for m in mentions
                        if not (m.canonical == "chair" and singularize(m.surface.lower()) == "seat")]

        seen: Set[str] = set()
        for m in mentions:
            m.first = m.canonical not in seen
            seen.add(m.canonical)
        return mentions

    def classes_in(self, text: str) -> Set[str]:
        return {m.canonical for m in self.extract(text)}


class CocoGroundTruth:
    """
    Ground-truth object sets per COCO image, built the CHAIR way:
    instance annotations plus objects named in the five reference captions.
    """

    def __init__(self, instances_file: str, captions_file: str,
                 extractor: Optional[ChairExtractor] = None):
        self.extractor = extractor or ChairExtractor()
        self.objects: Dict[int, Set[str]] = defaultdict(set)
        self.file_names: Dict[int, str] = {}
        self._load(instances_file, captions_file)

    def _load(self, instances_file: str, captions_file: str):
        with open(instances_file) as f:
            inst = json.load(f)
        cat_names = {c["id"]: c["name"] for c in inst["categories"]}
        unknown = set(cat_names.values()) - set(self.extractor.classes)
        if unknown:
            raise ValueError(f"COCO classes missing from synonym list: {unknown}")

        for img in inst["images"]:
            self.file_names[img["id"]] = img["file_name"]
        for ann in inst["annotations"]:
            self.objects[ann["image_id"]].add(cat_names[ann["category_id"]])

        with open(captions_file) as f:
            caps = json.load(f)
        for ann in caps["annotations"]:
            self.objects[ann["image_id"]] |= self.extractor.classes_in(ann["caption"])

    def image_ids(self) -> List[int]:
        return sorted(self.file_names)

    def label(self, image_id: int, mentions: List[Mention]) -> List[Mention]:
        """Set label = 1 for objects not in the image's ground truth."""
        gt = self.objects.get(image_id, set())
        for m in mentions:
            m.label = 0 if m.canonical in gt else 1
        return mentions
