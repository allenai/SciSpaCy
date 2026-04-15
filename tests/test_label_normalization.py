import unittest

import spacy
from spacy.tokens import Span

from scispacy.label_normalization import (
    LABEL_MAP,
    UNIFIED_LABELS,
    normalize_label,
    normalize_entities,
)


class TestNormalizeLabel(unittest.TestCase):
    def test_protein_maps_to_gene_or_gene_product(self):
        assert normalize_label("PROTEIN") == "GENE_OR_GENE_PRODUCT"

    def test_dna_maps_to_gene_or_gene_product(self):
        assert normalize_label("DNA") == "GENE_OR_GENE_PRODUCT"

    def test_ggp_maps_to_gene_or_gene_product(self):
        assert normalize_label("GGP") == "GENE_OR_GENE_PRODUCT"

    def test_chemical_labels(self):
        for label in ("CHEMICAL", "CHEBI", "SIMPLE_CHEMICAL", "AMINO_ACID"):
            assert normalize_label(label) == "CHEMICAL", f"{label} should map to CHEMICAL"

    def test_disease_labels(self):
        for label in ("DISEASE", "CANCER", "PATHOLOGICAL_FORMATION"):
            assert normalize_label(label) == "DISEASE", f"{label} should map to DISEASE"

    def test_cell_labels(self):
        for label in ("CELL", "CELL_TYPE", "CELL_LINE", "CL"):
            assert normalize_label(label) == "CELL", f"{label} should map to CELL"

    def test_organism_labels(self):
        assert normalize_label("ORGANISM") == "ORGANISM"
        assert normalize_label("TAXON") == "ORGANISM"

    def test_anatomy_labels(self):
        anatomy_labels = [
            "ANATOMICAL_SYSTEM", "CELLULAR_COMPONENT",
            "DEVELOPING_ANATOMICAL_STRUCTURE", "IMMATERIAL_ANATOMICAL_ENTITY",
            "MULTI-TISSUE_STRUCTURE", "ORGAN", "ORGANISM_SUBDIVISION",
            "ORGANISM_SUBSTANCE", "TISSUE",
        ]
        for label in anatomy_labels:
            assert normalize_label(label) == "ANATOMY", f"{label} should map to ANATOMY"

    def test_unknown_label_passes_through(self):
        assert normalize_label("MY_CUSTOM_LABEL") == "MY_CUSTOM_LABEL"
        assert normalize_label("ENTITY") == "ENTITY"

    def test_rna_labels(self):
        assert normalize_label("RNA") == "RNA"
        assert normalize_label("SO") == "RNA"

    def test_go_label(self):
        assert normalize_label("GO") == "BIOLOGICAL_PROCESS"

    def test_unified_labels_is_sorted(self):
        assert UNIFIED_LABELS == sorted(UNIFIED_LABELS)

    def test_all_map_values_in_unified_labels(self):
        for v in LABEL_MAP.values():
            assert v in UNIFIED_LABELS


class TestNormalizeEntities(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.blank("en")

    def test_entities_relabeled(self):
        doc = self.nlp("p53 binds to the androgen receptor in NSCLC cells")
        # manually set ents with model-specific labels
        doc.ents = [
            Span(doc, 0, 1, label="PROTEIN"),     # "p53"
            Span(doc, 5, 7, label="PROTEIN"),      # "androgen receptor"
            Span(doc, 8, 9, label="CELL_LINE"),    # "NSCLC"
        ]
        doc = normalize_entities(doc)

        labels = [ent.label_ for ent in doc.ents]
        assert labels == ["GENE_OR_GENE_PRODUCT", "GENE_OR_GENE_PRODUCT", "CELL"]

    def test_unknown_labels_preserved(self):
        doc = self.nlp("Something weird happened")
        doc.ents = [Span(doc, 0, 1, label="WEIRD")]
        doc = normalize_entities(doc)
        assert doc.ents[0].label_ == "WEIRD"

    def test_empty_ents(self):
        doc = self.nlp("no entities here")
        doc = normalize_entities(doc)
        assert len(doc.ents) == 0
