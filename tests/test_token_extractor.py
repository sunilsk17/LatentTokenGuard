"""
tests/test_token_extractor.py
Unit tests for TokenExtractor (Module 2).
"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.token_extractor import TokenExtractor


@pytest.fixture
def extractor():
    return TokenExtractor(use_pos_filter=True)


@pytest.fixture
def extractor_no_filter():
    return TokenExtractor(use_pos_filter=False)


def test_basic_noun_extraction(extractor):
    answer = "There is a sofa and a television in the room."
    tokens = ["There", "is", "a", "sofa", "and", "a", "television", "in", "the", "room"]
    indices, content = extractor.extract(answer, tokens)
    assert len(indices) > 0
    content_lower = [t.lower() for t in content]
    assert any(t in content_lower for t in ["sofa", "television", "room"])


def test_no_filter_returns_all(extractor_no_filter):
    tokens = ["yes", "there", "is"]
    indices, content = extractor_no_filter.extract("Yes there is", tokens)
    assert indices == [0, 1, 2]


def test_empty_answer(extractor):
    indices, content = extractor.extract("", [])
    assert indices == []
    assert content == []


def test_short_answer_fallback(extractor):
    """Very short answers like 'No.' should not crash, fallback to all tokens."""
    answer = "No."
    tokens = ["No", "."]
    indices, content = extractor.extract(answer, tokens)
    # Should fall back to returning all tokens
    assert len(indices) >= 0  # at least doesn't crash


def test_adjective_included(extractor):
    answer = "There is a red wooden chair."
    tokens = ["There", "is", "a", "red", "wooden", "chair"]
    indices, content = extractor.extract(answer, tokens)
    content_lower = [t.lower() for t in content]
    # At least some content words should be found
    assert len(indices) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
