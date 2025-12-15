from typing import Dict, Tuple, Optional
import os

import pytest
import spacy
import requests
from spacy.language import Language as SpacyModelType
from spacy.cli.download import download as spacy_download

from scispacy.custom_sentence_segmenter import pysbd_sentencizer
from scispacy.custom_tokenizer import combined_rule_tokenizer, combined_rule_prefixes, remove_new_lines
from scispacy.abbreviation import AbbreviationDetector

LOADED_SPACY_MODELS: Dict[Tuple[str, bool, bool, bool, bool, bool, Optional[bool]], SpacyModelType] = {}
MODEL_FALLBACKS = {"en_core_sci_sm": "en_core_web_sm"}


class MissingSpacyModel(RuntimeError):
    """Raised when neither the requested model nor its fallback are available."""


def _load_or_download_model(model_name: str, disable):
    """
    Try loading a spaCy model, downloading it first if needed. Raises
    MissingSpacyModel if a compatible wheel is unavailable.
    """
    try:
        return spacy.load(model_name, disable=disable)
    except OSError:
        print(f"Spacy model '{model_name}' not found.  Downloading and installing.")
        try:
            spacy_download(model_name)
        except (SystemExit, Exception) as exc:
            raise MissingSpacyModel(model_name) from exc
        return spacy.load(model_name, disable=disable)


def get_spacy_model(
    spacy_model_name: str,
    pos_tags: bool,
    parse: bool,
    ner: bool,
    with_custom_tokenizer: bool = False,
    with_sentence_segmenter: bool = False,
    with_serializable_abbreviation_detector: Optional[bool] = None,
) -> SpacyModelType:
    """
    In order to avoid loading spacy models repeatedly,
    we'll save references to them, keyed by the options
    we used to create the spacy model, so any particular
    configuration only gets loaded once.
    """
    options = (
        spacy_model_name,
        pos_tags,
        parse,
        ner,
        with_custom_tokenizer,
        with_sentence_segmenter,
        with_serializable_abbreviation_detector,
    )
    if options not in LOADED_SPACY_MODELS:
        disable = ["vectors", "textcat"]
        if not pos_tags:
            disable.append("tagger")
        if not parse:
            disable.append("parser")
        if not ner:
            disable.append("ner")
        try:
            spacy_model = _load_or_download_model(spacy_model_name, disable)
        except MissingSpacyModel:
            fallback_name = MODEL_FALLBACKS.get(spacy_model_name)
            if fallback_name is None:
                raise
            print(
                f"Falling back to spaCy model '{fallback_name}' because '{spacy_model_name}' "
                "is not available for this Python/spaCy version."
            )
            spacy_model = _load_or_download_model(fallback_name, disable)

        if with_custom_tokenizer:
            spacy_model.tokenizer = combined_rule_tokenizer(spacy_model)
        if with_sentence_segmenter:
            spacy_model.add_pipe("pysbd_sentencizer", first=True)
        if with_serializable_abbreviation_detector is not None:
            spacy_model.add_pipe("abbreviation_detector", config={"make_serializable": with_serializable_abbreviation_detector})

        LOADED_SPACY_MODELS[options] = spacy_model
    return LOADED_SPACY_MODELS[options]


@pytest.fixture()
def combined_rule_tokenizer_fixture():
    nlp = get_spacy_model("en_core_web_sm", True, True, True)
    tokenizer = combined_rule_tokenizer(nlp)
    return tokenizer


@pytest.fixture()
def en_with_combined_rule_tokenizer_fixture():
    nlp = get_spacy_model("en_core_web_sm", True, True, True, with_custom_tokenizer=True)
    return nlp


@pytest.fixture()
def en_with_combined_rule_tokenizer_and_segmenter_fixture():
    nlp = get_spacy_model("en_core_web_sm", True, True, True, with_custom_tokenizer=True, with_sentence_segmenter=True)
    return nlp


@pytest.fixture()
def test_data_fixtures_path():
    return os.path.join("tests", "custom_tests", "data_fixtures")


@pytest.fixture()
def test_raw_path():
    return os.path.join("tests", "custom_tests", "data_fixtures", "raw")


@pytest.fixture()
def test_pmids_path():
    return os.path.join("tests", "custom_tests", "data_fixtures", "test.pmids")


@pytest.fixture()
def test_conll_path():
    return os.path.join("tests", "custom_tests", "data_fixtures", "test.conllu")


@pytest.fixture()
def test_model_dir():
    return os.path.join("tests", "custom_tests", "data_fixtures", "tmp_model_dir")


@pytest.fixture()
def combined_all_model_fixture():
    nlp = get_spacy_model("en_core_sci_sm", True, True, True, with_custom_tokenizer=True, with_sentence_segmenter=False, with_serializable_abbreviation_detector=True)
    return nlp

@pytest.fixture()
def combined_all_model_fixture_non_serializable_abbrev():
    nlp = get_spacy_model("en_core_sci_sm", True, True, True, with_custom_tokenizer=True, with_sentence_segmenter=False, with_serializable_abbreviation_detector=False)
    return nlp

@pytest.fixture()
def combined_rule_prefixes_fixture():
    return combined_rule_prefixes()


@pytest.fixture()
def remove_new_lines_fixture():
    return remove_new_lines


@pytest.fixture()
def default_en_tokenizer_fixture():
    nlp = get_spacy_model("en_core_web_sm", True, True, True)
    return nlp.tokenizer


@pytest.fixture()
def default_en_model_fixture():
    nlp = get_spacy_model("en_core_web_sm", True, True, True)
    return nlp
