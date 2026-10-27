"""
tests/test_caption_bench.py
Unit tests for the caption benchmark: CHAIR extraction, token mapping and evaluation.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from caption_bench.capture import token_at_char
from caption_bench.chair import ChairExtractor, CocoGroundTruth, singularize
from caption_bench.evaluate import (crossfit_evaluate, crossfit_scores, evaluate_run, fast_auroc,
                                    load_run, stratified_auroc)
from sklearn.metrics import roc_auc_score


@pytest.fixture(scope="module")
def chair():
    return ChairExtractor()


def classes(chair, text):
    return [m.canonical for m in chair.extract(text)]


def test_all_80_classes_loaded(chair):
    assert len(chair.classes) == 80
    assert "dining table" in chair.classes and "hair drier" in chair.classes


def test_synonyms_and_plurals(chair):
    assert classes(chair, "Two puppies and three kittens.") == ["dog", "cat"]
    assert classes(chair, "A man and some children near the buses.") == ["person", "person", "bus"]
    assert classes(chair, "Knives and wine glasses on a table.") == ["knife", "wine glass", "dining table"]


def test_multi_word_objects_beat_single_words(chair):
    assert classes(chair, "A teddy bear next to a hot dog.") == ["teddy bear", "hot dog"]
    assert classes(chair, "a cell phone and a traffic light") == ["cell phone", "traffic light"]


def test_chair_phrase_rules(chair):
    assert classes(chair, "A baby elephant walks.") == ["elephant"]
    assert classes(chair, "A baby animal sleeps.") == []
    assert classes(chair, "A passenger jet takes off.") == ["airplane"]
    assert classes(chair, "The toilet seat is up.") == ["toilet"]
    assert classes(chair, "A toilet next to a seat.") == ["toilet"]
    assert classes(chair, "A man in a bow tie.") == ["person", "tie"]


def test_spans_and_first_flag(chair):
    text = "A dog chases another Dog."
    ms = chair.extract(text)
    assert [text[m.start:m.end] for m in ms] == ["dog", "Dog"]
    assert [m.first for m in ms] == [True, False]


def test_singularize():
    assert singularize("benches") == "bench"
    assert singularize("bus") == "bus"
    assert singularize("giraffes") == "giraffe"
    assert singularize("glass") == "glass"


def test_ground_truth_labels(chair, tmp_path):
    inst = {"images": [{"id": 1, "file_name": "a.jpg"}],
            "categories": [{"id": i, "name": c} for i, c in enumerate(chair.classes)],
            "annotations": [{"image_id": 1, "category_id": chair.classes.index("dog")}]}
    caps = {"annotations": [{"image_id": 1, "caption": "A dog on a couch."}]}
    (tmp_path / "inst.json").write_text(json.dumps(inst))
    (tmp_path / "caps.json").write_text(json.dumps(caps))
    gt = CocoGroundTruth(str(tmp_path / "inst.json"), str(tmp_path / "caps.json"), chair)
    ms = gt.label(1, chair.extract("A puppy on a sofa near a cat."))
    assert [(m.canonical, m.label) for m in ms] == [("dog", 0), ("couch", 0), ("cat", 1)]


def test_token_at_char():
    # caption "A dog runs" tokenized as ["A", " dog", " runs"]
    starts = [0, 1, 5]
    assert token_at_char(starts, 2) == 1      # 'd' of dog
    assert token_at_char(starts, 6) == 2      # 'r' of runs
    assert token_at_char([0, 0, 3], 0) == 1   # empty special token is skipped


def test_fast_auroc_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = rng.normal(size=(200, 3, 2)) + y[:, None, None]
    fast = fast_auroc(y, s)
    for i in range(3):
        for j in range(2):
            assert fast[i, j] == pytest.approx(roc_auc_score(y, s[:, i, j]))


def test_stratified_auroc():
    labels = np.array([0, 1, 0, 1, 0, 0, 1, 1])
    scores = np.array([0, 1, 1, 0, 5, 6, 7, 8], dtype=float)
    strata = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    expected = (4 * roc_auc_score(labels[:4], scores[:4]) + 4 * 1.0) / 8
    assert float(stratified_auroc(labels, scores, strata)) == pytest.approx(expected)
    # a stratum with one class only is skipped
    strata_one = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    assert float(stratified_auroc(labels, scores, strata_one)) == pytest.approx(roc_auc_score(labels[:4], scores[:4]))


def _fake_run(path, n_images=60, layers=4, seed=0, position_only=False, hidden_dim=0, glsim=False):
    """
    A run where layer pair (2, 1) 'max' separates hallucinations. With position_only,
    labels and that feature both follow the caption position and nothing else.
    With hidden_dim, also writes hidden/ where layer 3 carries the label along one direction.
    With glsim, also writes glsim/ where the K=8 local score at object layer 3, image layer 2
    is lower for hallucinations.
    """
    rng = np.random.default_rng(seed)
    pos_rng = np.random.default_rng(seed + 1)
    stat_names = ["max", "top4", "top8", "top16", "top32", "mean", "std", "global"]
    os.makedirs(path / "feats")
    if hidden_dim:
        os.makedirs(path / "hidden")
    if glsim:
        os.makedirs(path / "glsim")
    (path / "run.json").write_text(json.dumps({"stat_names": stat_names,
                                               "model": {"model_id": "fake"}}))
    with open(path / "captions.jsonl", "w") as f:
        for img in range(n_images):
            m = int(rng.integers(0, 5))
            positions = pos_rng.integers(0, 100, m)
            labels = (rng.random(m) < positions / 100) if position_only else rng.integers(0, 2, m)
            mentions = [{"canonical": "dog", "label": int(l), "first": True} for l in labels]
            f.write(json.dumps({"image_id": img, "caption": "", "mentions": mentions}) + "\n")
            if m == 0:
                continue
            grid = rng.uniform(0.2, 0.4, (m, layers, layers, len(stat_names)))
            grid[..., 6] = 0.05
            grid[:, 2, 1, 0] -= 0.01 * positions if position_only else 0.3 * labels
            np.savez(path / "feats" / f"{img}.npz", token_index=positions.astype(np.int32),
                     nll=rng.normal(size=m).astype(np.float32),
                     entropy=rng.normal(size=m).astype(np.float32),
                     ic=rng.uniform(size=(m, layers)).astype(np.float32),
                     grid_at=grid.astype(np.float32), grid_pre=grid.astype(np.float32))
            if hidden_dim:
                at = rng.normal(size=(m, layers, hidden_dim))
                at[:, 3, 0] += 2.0 * labels
                d = np.arange(layers)
                np.savez(path / "hidden" / f"{img}.npz", token_index=positions.astype(np.int32),
                         at=at.astype(np.float16), check_max=grid[:, d, d, 0].astype(np.float32))
            if glsim:
                local = rng.uniform(0.2, 0.4, (m, layers, layers, 4))
                local[:, 3, 2, 1] -= 0.3 * labels
                np.savez(path / "glsim" / f"{img}.npz", token_index=positions.astype(np.int32),
                         local=local.astype(np.float32),
                         **{"global": rng.uniform(0.2, 0.4, (m, layers, layers)).astype(np.float32)})


def test_evaluate_run_finds_planted_layer(tmp_path):
    _fake_run(tmp_path)
    table = load_run(str(tmp_path))
    cfg = {"calib_fraction": 0.5, "split_seed": 0, "bootstrap": 50, "keep_real": 0.95}
    res = evaluate_run(table, cfg)
    sel = res["detectors"]["Max-patch cosine, layer pair"]["selection"]
    assert (sel["text_layer"], sel["image_layer"], sel["sign"]) == (2, 1, "-")
    assert res["detectors"]["Max-patch cosine, layer pair"]["auroc"] > 0.9
    assert 0.3 < res["detectors"]["NLL"]["auroc"] < 0.7
    rm = res["detectors"]["Max-patch cosine, layer pair"]["remove"]
    assert rm["after"]["CHAIR_i"] < rm["before"]["CHAIR_i"]
    # the planted signal does not depend on position, so the controls keep it
    assert res["detectors"]["Max-patch cosine, layer pair"]["auroc_pos"] > 0.85
    assert res["detectors"]["Best latent stat, position-corrected"]["auroc"] > 0.85
    assert 0.3 < res["detectors"]["Token position"]["auroc"] < 0.7


def test_position_controls_remove_a_pure_position_signal(tmp_path):
    _fake_run(tmp_path, n_images=400, position_only=True)
    cfg = {"calib_fraction": 0.5, "split_seed": 0, "bootstrap": 20, "keep_real": 0.95,
           "position_bins": 10}
    res = evaluate_run(load_run(str(tmp_path)), cfg)["detectors"]
    assert res["Token position"]["auroc"] > 0.75
    assert res["Max-patch cosine, layer pair"]["auroc"] > 0.75
    assert abs(res["Max-patch cosine, layer pair"]["auroc_pos"] - 0.5) < 0.1
    assert abs(res["Best latent stat, position-corrected"]["auroc"] - 0.5) < 0.1


CV_CFG = {"calib_fraction": 0.5, "split_seed": 0, "bootstrap": 50, "keep_real": 0.95,
          "position_bins": 5, "cv_folds": 4}


def test_crossfit_scores_every_mention_once(tmp_path):
    _fake_run(tmp_path, n_images=80)
    table = load_run(str(tmp_path))
    scores, strata, selections = crossfit_scores(table, folds=4, seed=0, position_bins=5)
    s = scores["Max-patch cosine, layer pair"]
    assert s.shape == table.labels.shape and np.all((s >= 0) & (s <= 1))
    assert len(selections["Max-patch cosine, layer pair"]) == 4
    assert set(np.unique(strata)) <= set(range(5))


def test_crossfit_finds_planted_signal_and_compares(tmp_path):
    _fake_run(tmp_path, n_images=120)
    cv = crossfit_evaluate(load_run(str(tmp_path)), CV_CFG)
    assert cv["detectors"]["Best latent stat (at)"]["auroc"] > 0.9
    pairs = {(c["detector"], c["baseline"]): c for c in cv["comparisons"]}
    nll = pairs["Best latent stat (at)", "NLL"]
    assert nll["overall"]["diff"] > 0.2 and nll["overall"]["p_le_0"] == 0.0
    assert nll["within_position"]["ci"][0] > 0
    fused = pairs["Fusion: NLL + best latent", "NLL"]
    assert fused["overall"]["diff"] > 0 and fused["within_position"]["ci"][0] > 0


def test_crossfit_position_only_signal_vanishes_within_bins(tmp_path):
    _fake_run(tmp_path, n_images=400, position_only=True)
    cv = crossfit_evaluate(load_run(str(tmp_path)), {**CV_CFG, "position_bins": 10})
    assert cv["detectors"]["Best latent stat (at)"]["auroc"] > 0.75
    assert abs(cv["detectors"]["Best latent stat (at)"]["auroc_pos"] - 0.5) < 0.1


def test_linear_probe_on_hidden_states(tmp_path):
    _fake_run(tmp_path, n_images=200, hidden_dim=16)
    table = load_run(str(tmp_path))
    assert table.hidden.shape[1:] == (4, 16)
    assert table.hidden_check["max_abs_diff"] < 1e-6
    cfg = {"calib_fraction": 0.5, "split_seed": 0, "bootstrap": 20, "keep_real": 0.95}
    res = evaluate_run(table, cfg)
    probe = res["detectors"]["Linear probe on hidden states"]
    assert probe["selection"]["layer"] == 3 and probe["auroc"] > 0.8
    cv = crossfit_evaluate(table, CV_CFG)
    pairs = {(c["detector"], c["baseline"]) for c in cv["comparisons"]}
    assert ("Best latent stat (at)", "Linear probe on hidden states") in pairs


def test_runs_without_hidden_states_skip_the_probe(tmp_path):
    _fake_run(tmp_path, n_images=80)
    table = load_run(str(tmp_path))
    assert table.hidden is None
    cv = crossfit_evaluate(table, CV_CFG)
    assert "Linear probe on hidden states" not in cv["detectors"]
    assert len(cv["comparisons"]) == 3


def test_glsim_detector_finds_planted_setting(tmp_path):
    _fake_run(tmp_path, n_images=150, glsim=True)
    table = load_run(str(tmp_path))
    assert table.glsim_local.shape[1:] == (4, 4, 4)
    cfg = {"calib_fraction": 0.5, "split_seed": 0, "bootstrap": 20, "keep_real": 0.95}
    res = evaluate_run(table, cfg)["detectors"]
    assert "GLSim (published settings)" not in res          # fake model id has no published setting
    sel = res["GLSim (settings chosen on calibration)"]["selection"]
    assert (sel["object_layer"], sel["image_layer"], sel["K"]) == (3, 2, 8) and sel["w"] < 0.5
    assert res["GLSim (settings chosen on calibration)"]["auroc"] > 0.9
    cv = crossfit_evaluate(table, CV_CFG)
    pairs = {(c["detector"], c["baseline"]) for c in cv["comparisons"]}
    assert ("Best latent stat (at)", "GLSim (settings chosen on calibration)") in pairs
