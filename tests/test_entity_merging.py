import unittest

import spacy
from spacy.tokens import Doc, Span

from scispacy.entity_merging import merge_overlapping_spans


class TestMergeOverlappingSpans(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.blank("en")

    def _make_doc(self, text):
        return self.nlp(text)

    def test_no_spans_returns_empty(self):
        doc = self._make_doc("hello world")
        result = merge_overlapping_spans([], doc)
        assert result == []

    def test_non_overlapping_spans_kept(self):
        doc = self._make_doc("Spinal atrophy and motor neuron disease are conditions")
        span_a = doc.char_span(0, 14, label="ENTITY")   # "Spinal atrophy"
        span_b = doc.char_span(19, 39, label="ENTITY")  # "motor neuron disease"
        assert span_a is not None
        assert span_b is not None
        result = merge_overlapping_spans([span_a, span_b], doc)
        assert len(result) == 2

    def test_overlapping_spans_keep_longest(self):
        doc = self._make_doc("Spinal and bulbar muscular atrophy is a disease")
        short = doc.char_span(0, 6, label="ENTITY")   # "Spinal"
        long = doc.char_span(0, 34, label="ENTITY")   # "Spinal and bulbar muscular atrophy"
        assert short is not None
        assert long is not None
        result = merge_overlapping_spans([short, long], doc)
        assert len(result) == 1
        assert result[0].text == "Spinal and bulbar muscular atrophy"

    def test_partial_overlap_keeps_longest(self):
        doc = self._make_doc("bulbar muscular atrophy is studied")
        span_a = doc.char_span(0, 23, label="ENTITY")  # "bulbar muscular atrophy"
        span_b = doc.char_span(7, 23, label="ENTITY")  # "muscular atrophy"
        assert span_a is not None
        assert span_b is not None
        result = merge_overlapping_spans([span_a, span_b], doc)
        assert len(result) == 1
        assert result[0].text == "bulbar muscular atrophy"

    def test_duplicate_spans_deduplicated(self):
        doc = self._make_doc("motor neuron disease is common")
        span_a = doc.char_span(0, 20, label="ENTITY")
        span_b = doc.char_span(0, 20, label="ENTITY")
        assert span_a is not None
        assert span_b is not None
        result = merge_overlapping_spans([span_a, span_b], doc)
        assert len(result) == 1

    def test_many_overlapping_spans(self):
        # Simulates entities from multiple models with different granularity
        doc = self._make_doc("Spinal and bulbar muscular atrophy caused by androgen receptor")
        spans = []
        # model A: fragments
        spans.append(doc.char_span(0, 6, label="ENTITY"))    # "Spinal"
        spans.append(doc.char_span(11, 34, label="ENTITY"))  # "bulbar muscular atrophy"
        # model B: full phrase
        spans.append(doc.char_span(0, 34, label="ENTITY"))   # "Spinal and bulbar muscular atrophy"
        # model C: second entity
        spans.append(doc.char_span(45, 62, label="ENTITY"))  # "androgen receptor"
        # filter out any None from char_span misalignment
        spans = [s for s in spans if s is not None]

        result = merge_overlapping_spans(spans, doc)
        texts = {s.text for s in result}
        assert "Spinal and bulbar muscular atrophy" in texts
        assert "androgen receptor" in texts
        # fragments should be gone
        assert "Spinal" not in texts
        assert "bulbar muscular atrophy" not in texts
