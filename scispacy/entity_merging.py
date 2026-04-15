"""
Merge entities recognized by different NER models,
optionally incorporating abbreviation long forms as entities.

Usage
-----

.. code-block:: python

    import spacy
    from scispacy.entity_merging import merge_entities

    text = "Spinal and bulbar muscular atrophy (SBMA) is an inherited motor neuron disease."
    doc = merge_entities(
        text,
        model_names=["en_core_sci_sm", "en_core_sci_lg"],
        use_abbreviations=True,
    )
    print(doc.ents)

Or as a function you call on an already-processed list of docs:

.. code-block:: python

    from scispacy.entity_merging import merge_overlapping_spans

    merged = merge_overlapping_spans(all_spans, doc)
"""

from typing import List, Optional

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.util import filter_spans


def merge_overlapping_spans(spans: List[Span], doc: Doc) -> List[Span]:
    """
    Given a flat list of (possibly overlapping) spans that all reference
    the same Doc, return a filtered list keeping the longest non-overlapping
    spans.  Ties are broken by whichever span appears first.
    """
    if not spans:
        return []
    # filter_spans keeps the longest span when there's overlap
    return filter_spans(spans)


def _collect_entity_spans(
    text: str,
    model_names: List[str],
    use_abbreviations: bool,
) -> tuple:
    """
    Run *text* through each model in *model_names*, collect every entity
    span, and (optionally) add spans for abbreviation long forms.

    Returns (base_doc, all_spans) where *base_doc* is the Doc produced by
    the first model and *all_spans* are Span objects that all reference
    *base_doc*.
    """
    if not model_names:
        raise ValueError("model_names must contain at least one model")

    pipelines = [spacy.load(name) for name in model_names]
    base_nlp = pipelines[0]
    base_doc = base_nlp(text)

    all_spans: List[Span] = list(base_doc.ents)

    # Entities from the remaining models need to be projected onto base_doc
    # because spaCy Spans are tied to a specific Doc object.
    for nlp in pipelines[1:]:
        other_doc = nlp(text)
        for ent in other_doc.ents:
            try:
                span = base_doc.char_span(ent.start_char, ent.end_char, label=ent.label_)
            except Exception:
                continue
            if span is not None:
                all_spans.append(span)

    if use_abbreviations:
        all_spans = _add_abbreviation_spans(base_nlp, base_doc, all_spans)

    return base_doc, all_spans


def _add_abbreviation_spans(
    nlp: Language, doc: Doc, spans: List[Span]
) -> List[Span]:
    """
    If AbbreviationDetector is in the pipeline, use detected long forms
    to create additional entity spans.
    """
    try:
        nlp.get_pipe("abbreviation_detector")
    except KeyError:
        # no abbreviation detector in this pipeline — add one temporarily
        from scispacy.abbreviation import AbbreviationDetector  # noqa: F811

        nlp.add_pipe("abbreviation_detector")
        doc = nlp(doc.text)

    for abrv in doc._.abbreviations:
        long_form = abrv._.long_form
        if long_form is None:
            continue
        # long_form is already a Span on our doc
        if isinstance(long_form, Span):
            spans.append(long_form)

    return spans


def merge_entities(
    text: str,
    model_names: List[str],
    use_abbreviations: bool = True,
) -> Doc:
    """
    Run *text* through multiple spaCy NER models, collect all recognized
    entities, optionally add abbreviation long forms, and return a single
    Doc whose ``.ents`` contains the longest non-overlapping entity spans.

    Parameters
    ----------
    text : str
        The text to process.
    model_names : list of str
        Names of spaCy models to use (e.g. ``["en_core_sci_sm", "en_core_sci_lg"]``).
    use_abbreviations : bool, optional (default True)
        Whether to incorporate abbreviation long forms as candidate entities.

    Returns
    -------
    Doc
        A spaCy Doc with merged entities set as ``doc.ents``.
    """
    base_doc, all_spans = _collect_entity_spans(text, model_names, use_abbreviations)
    merged = merge_overlapping_spans(all_spans, base_doc)
    base_doc.ents = merged
    return base_doc
