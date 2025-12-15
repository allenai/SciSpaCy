import multiprocessing as mp
import os
import shutil
import sys

import pytest
import spacy
from spacy.vocab import Vocab

if mp.get_start_method(allow_none=True) != "spawn":  # pragma: no cover - import-time behavior
    mp.set_start_method("spawn", force=True)


def test_custom_segmentation(combined_all_model_fixture):
    text = "Induction of cytokine expression in leukocytes by binding of thrombin-stimulated platelets. BACKGROUND: Activated platelets tether and activate myeloid leukocytes."
    doc = combined_all_model_fixture(text)
    sents = list(doc.sents)
    assert len(sents) == 2
    expected_tokens = [
        "Induction",
        "of",
        "cytokine",
        "expression",
        "in",
        "leukocytes",
        "by",
        "binding",
        "of",
        "thrombin-stimulated",
        "platelets",
        ".",
        "BACKGROUND",
        ":",
        "Activated",
        "platelets",
        "tether",
        "and",
        "activate",
        "myeloid",
        "leukocytes",
        ".",
    ]
    actual_tokens = [t.text for t in doc]
    assert expected_tokens == actual_tokens
    assert doc.has_annotation("DEP")
    assert doc[0].dep_ == "ROOT"
    assert doc[0].tag_ == "NN"

def test_full_pipe_serializable(combined_all_model_fixture):
    text = "Induction of cytokine expression in leukocytes (CEIL) by binding of thrombin-stimulated platelets. BACKGROUND: Activated platelets tether and activate myeloid leukocytes."
    doc = [doc for doc in combined_all_model_fixture.pipe([text, text], n_process = 2)][0]
    # If we got here this means that both model is serializable and there is an abbreviation that would break if it wasn't
    assert len(doc._.abbreviations) > 0
    abbrev = doc._.abbreviations[0]
    assert abbrev["short_text"] == "CEIL"
    assert abbrev["long_text"] == "cytokine expression in leukocytes"
    assert doc[abbrev["short_start"] : abbrev["short_end"]].text == abbrev["short_text"]
    assert doc[abbrev["long_start"] : abbrev["long_end"]].text == abbrev["long_text"]

def test_full_pipe_not_serializable(combined_all_model_fixture_non_serializable_abbrev):
    text = "Induction of cytokine expression in leukocytes (CEIL) by binding of thrombin-stimulated platelets. BACKGROUND: Activated platelets tether and activate myeloid leukocytes."
    # This line requires the pipeline to be serializable, so the test should fail here
    doc = combined_all_model_fixture_non_serializable_abbrev(text)
    with pytest.raises(TypeError):
        doc.to_bytes()

# Below is the test version to be used once we move to spacy v3.1.0 or higher
# def test_full_pipe_not_serializable(combined_all_model_fixture_non_serializable_abbrev):
#     text = "Induction of cytokine expression in leukocytes (CEIL) by binding of thrombin-stimulated platelets. BACKGROUND: Activated platelets tether and activate myeloid leukocytes."
#     # This line requires the pipeline to be serializable (because it uses 2 processes), so the test should fail here
#     with pytest.raises(TypeError):
#         list(combined_all_model_fixture_non_serializable_abbrev.pipe([text, text], n_process = 2))
