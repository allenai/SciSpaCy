"""
Normalize NER entity labels across different scispaCy models.

Each scispaCy NER model is trained on a different corpus, so the same
biomedical concept can get different labels depending on which model
produced it.  For example:

- JNLPBA calls genes ``PROTEIN`` or ``DNA``
- BIONLP13CG calls them ``GENE_OR_GENE_PRODUCT``
- CRAFT calls them ``GGP``

This module provides a mapping from every model-specific label to a
small set of **unified labels**, making it practical to merge entities
from multiple models (see :mod:`scispacy.entity_merging`).

Usage
-----

.. code-block:: python

    from scispacy.label_normalization import normalize_label

    normalize_label("PROTEIN")            # -> "GENE_OR_GENE_PRODUCT"
    normalize_label("GGP")                # -> "GENE_OR_GENE_PRODUCT"
    normalize_label("CHEBI")              # -> "CHEMICAL"
    normalize_label("DISEASE")            # -> "DISEASE"
    normalize_label("NONEXISTENT_LABEL")  # -> "NONEXISTENT_LABEL" (unchanged)
"""

from typing import Dict, List, Optional
from spacy.tokens import Doc, Span

# ---------------------------------------------------------------------------
# Unified label set and mappings
# ---------------------------------------------------------------------------
#
# The unified labels are deliberately broad.  The goal is to make spans
# from different models comparable, *not* to create a perfect ontology.
#
# Sources for each corpus's label set:
#   CRAFT:       GGP, SO, TAXON, CHEBI, GO, CL
#   JNLPBA:      DNA, CELL_TYPE, CELL_LINE, RNA, PROTEIN
#   BC5CDR:      DISEASE, CHEMICAL
#   BIONLP13CG:  AMINO_ACID, ANATOMICAL_SYSTEM, CANCER, CELL,
#                 CELLULAR_COMPONENT, DEVELOPING_ANATOMICAL_STRUCTURE,
#                 GENE_OR_GENE_PRODUCT, IMMATERIAL_ANATOMICAL_ENTITY,
#                 MULTI-TISSUE_STRUCTURE, ORGAN, ORGANISM,
#                 ORGANISM_SUBDIVISION, ORGANISM_SUBSTANCE,
#                 PATHOLOGICAL_FORMATION, SIMPLE_CHEMICAL, TISSUE

LABEL_MAP: Dict[str, str] = {
    # --- gene / protein ---
    "PROTEIN": "GENE_OR_GENE_PRODUCT",
    "DNA": "GENE_OR_GENE_PRODUCT",
    "GGP": "GENE_OR_GENE_PRODUCT",
    "GENE_OR_GENE_PRODUCT": "GENE_OR_GENE_PRODUCT",

    # --- RNA ---
    "RNA": "RNA",
    "SO": "RNA",  # CRAFT "sequence ontology" — mostly RNA/gene features

    # --- chemical / drug ---
    "CHEMICAL": "CHEMICAL",
    "CHEBI": "CHEMICAL",
    "SIMPLE_CHEMICAL": "CHEMICAL",
    "AMINO_ACID": "CHEMICAL",

    # --- disease / pathology ---
    "DISEASE": "DISEASE",
    "CANCER": "DISEASE",
    "PATHOLOGICAL_FORMATION": "DISEASE",

    # --- cell ---
    "CELL": "CELL",
    "CELL_TYPE": "CELL",
    "CELL_LINE": "CELL",
    "CL": "CELL",  # CRAFT cell ontology

    # --- organism ---
    "ORGANISM": "ORGANISM",
    "TAXON": "ORGANISM",

    # --- anatomy ---
    "ANATOMICAL_SYSTEM": "ANATOMY",
    "CELLULAR_COMPONENT": "ANATOMY",
    "DEVELOPING_ANATOMICAL_STRUCTURE": "ANATOMY",
    "IMMATERIAL_ANATOMICAL_ENTITY": "ANATOMY",
    "MULTI-TISSUE_STRUCTURE": "ANATOMY",
    "ORGAN": "ANATOMY",
    "ORGANISM_SUBDIVISION": "ANATOMY",
    "ORGANISM_SUBSTANCE": "ANATOMY",
    "TISSUE": "ANATOMY",

    # --- biological process ---
    "GO": "BIOLOGICAL_PROCESS",
}

# The set of unified labels a user can expect after normalization.
UNIFIED_LABELS = sorted({v for v in LABEL_MAP.values()})


def normalize_label(label: str) -> str:
    """
    Map a model-specific NER label to the unified label set.

    If the label isn't in the mapping it is returned as-is, so custom
    labels from user-trained models still pass through.
    """
    return LABEL_MAP.get(label, label)


def normalize_entities(doc: Doc) -> Doc:
    """
    Replace entity labels in *doc* with their unified equivalents.

    Operates in-place and returns the same Doc for convenience.
    """
    new_ents: List[Span] = []
    for ent in doc.ents:
        unified = normalize_label(ent.label_)
        new_ents.append(
            Span(doc, ent.start, ent.end, label=unified)
        )
    doc.ents = new_ents
    return doc
