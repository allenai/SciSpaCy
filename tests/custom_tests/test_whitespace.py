# coding: utf-8
"""Test that tokens are created correctly for whitespace."""


from __future__ import unicode_literals

import importlib.util

import pytest

import spacy
from spacy.language import Language as SpacyModelType

from scispacy.custom_sentence_segmenter import pysbd_sentencizer


_MODEL_NAME = "en_core_sci_sm"
if importlib.util.find_spec(_MODEL_NAME) is None:  # pragma: no cover - env specific
    pytest.skip(
        f"{_MODEL_NAME} is required for whitespace tests; install the model to run them.",
        allow_module_level=True,
    )
_shared_nlp = spacy.load(_MODEL_NAME)


class TestWhitespace:
    nlp = _shared_nlp

    @pytest.mark.parametrize("text", ["lorem ipsum"])
    def test_tokenizer_splits_single_space(self, text):
        tokens = self.nlp(text)
        assert len(tokens) == 2

    @pytest.mark.parametrize("text", ["lorem  ipsum"])
    def test_tokenizer_splits_double_space(self, text):
        tokens = self.nlp(text)
        assert len(tokens) == 3
        assert tokens[1].text == " "

    @pytest.mark.parametrize("text", ["lorem ipsum  "])
    def test_tokenizer_handles_double_trainling_ws(self, text):
        tokens = self.nlp(text)
        assert repr(tokens.text_with_ws) == repr(text)

    @pytest.mark.parametrize("text", ["lorem\nipsum"])
    def test_tokenizer_splits_newline(self, text):
        tokens = self.nlp(text)
        assert len(tokens) == 3
        assert tokens[1].text == "\n"

    @pytest.mark.parametrize("text", ["lorem \nipsum"])
    def test_tokenizer_splits_newline_space(self, text):
        tokens = self.nlp(text)
        assert len(tokens) == 3

    @pytest.mark.parametrize("text", ["lorem  \nipsum"])
    def test_tokenizer_splits_newline_double_space(self, text):
        tokens = self.nlp(text)
        assert len(tokens) == 3

    @pytest.mark.parametrize("text", ["lorem \n ipsum"])
    def test_tokenizer_splits_newline_space_wrap(self, text):
        tokens = self.nlp(text)
        assert len(tokens) == 3
