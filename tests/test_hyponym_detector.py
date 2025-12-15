# pylint: disable=no-self-use,invalid-name
import unittest

from scispacy.hyponym_detector import HyponymDetector
from tests.conftest import get_spacy_model


class TestHyponymDetector(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.nlp = get_spacy_model("en_core_sci_sm", True, True, True)
        self.detector = HyponymDetector(self.nlp, extended=True)
        self.nlp.add_pipe("hyponym_detector", config={"extended": True}, last=True)

    def tearDown(self):
        if "hyponym_detector" in self.nlp.pipe_names:
            self.nlp.remove_pipe("hyponym_detector")
        super().tearDown()

    def test_sentences(self):
        text = (
            "Recognizing that the preferred habitats for the species "
            "are in the valleys, systematic planting of keystone plant "
            "species such as fig trees (Ficus) creates the best microhabitats."
        )
        doc = self.nlp(text)
        fig_trees = doc[21:23]
        plant_species = doc[16:19]
        assert doc._.hearst_patterns
        predicate, hypernym, hyponym = doc._.hearst_patterns[0]
        assert predicate == "such_as"
        assert hyponym == fig_trees
        # Different models may tag "Keystone" as a noun or adjective; accept either span.
        assert hypernym.text in {"keystone plant species", "plant species"}

        doc = self.nlp("SARS, or other coronaviruses, are bad.")
        assert doc._.hearst_patterns == [("other", doc[4:5], doc[0:1])]
        doc = self.nlp("Coronaviruses, including SARS and MERS, are bad.")
        assert doc._.hearst_patterns == [
            ("include", doc[0:1], doc[3:4]),
            ("include", doc[0:1], doc[5:6]),
        ]

    def test_find_noun_compound_head(self):

        doc = self.nlp("The potassium channel is good.")

        head = self.detector.find_noun_compound_head(doc[1])
        assert head == doc[2]

        doc = self.nlp("Planting of large plants.")
        head = self.detector.find_noun_compound_head(doc[3])
        # Planting is a noun, but not a compound with 'plants'.
        assert head != doc[0]
        assert head == doc[3]

    def test_expand_noun_phrase(self):
        doc = self.nlp("Keystone plant habitats are good.")
        chunk = self.detector.expand_to_noun_compound(doc[1], doc)
        assert chunk.end == 3
        assert chunk.start in (0, 1)
