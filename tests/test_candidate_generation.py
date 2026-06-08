import os
import unittest
import tempfile

from scispacy.candidate_generation import CandidateGenerator, create_tfidf_ann_index, MentionCandidate, load_approximate_nearest_neighbours_index, LinkerPaths
from scispacy.umls_utils import UmlsKnowledgeBase


class TestCandidateGeneration(unittest.TestCase):

    def setUp(self):
        super().setUp()

    def test_create_index(self):

        umls_fixture = UmlsKnowledgeBase("tests/fixtures/umls_test_fixture.json")
        with tempfile.TemporaryDirectory() as dir_name:
            umls_concept_aliases, tfidf_vectorizer, ann_index = create_tfidf_ann_index(dir_name, umls_fixture)

        assert len(umls_concept_aliases) == 93
        assert len(ann_index) == 93  # Number of deduplicated aliases + canonical ids
        tfidf_params = tfidf_vectorizer.get_params()

        assert tfidf_params["analyzer"] == "char_wb"
        assert tfidf_params["min_df"] == 10
        assert tfidf_params["ngram_range"] == (3, 3)

    def test_candidate_generation(self):

        umls_fixture = UmlsKnowledgeBase("tests/fixtures/umls_test_fixture.json")
        with tempfile.TemporaryDirectory() as dir_name:
            umls_concept_aliases, tfidf_vectorizer, ann_index = create_tfidf_ann_index(dir_name, umls_fixture)

        candidate_generator = CandidateGenerator(ann_index, tfidf_vectorizer, umls_concept_aliases, umls_fixture)
        results = candidate_generator(['(131)I-Macroaggregated Albumin'], 10)

        canonical_ids = [x.concept_id for x in results[0]]
        assert canonical_ids == ['C0000005', 'C0000015', 'C0000074', 'C0000102', 'C0000103']

        # The mention was an exact match, so should have a distance of zero to a concept:
        assert results[0][0] == MentionCandidate(concept_id='C0000005',
                                                 aliases=['(131)I-Macroaggregated Albumin'],
                                                 similarities=[1.0])

        # Test we don't crash with zero vectors
        results = candidate_generator(['ZZZZ'], 10)
        assert results == [[]]

    def test_empty_list(self):

        umls_fixture = UmlsKnowledgeBase("tests/fixtures/umls_test_fixture.json")
        with tempfile.TemporaryDirectory() as dir_name:
            umls_concept_aliases, tfidf_vectorizer, ann_index = create_tfidf_ann_index(dir_name, umls_fixture)

        candidate_generator = CandidateGenerator(ann_index, tfidf_vectorizer, umls_concept_aliases, umls_fixture)
        results = candidate_generator([], 10)

        assert results == []


    def test_load_npz_corrupt_file_raises_useful_error(self):
        with tempfile.TemporaryDirectory() as dir_name:
            corrupt_file = os.path.join(dir_name, "tfidf_vectors_sparse.npz")
            with open(corrupt_file, "wb") as f:
                f.write(b"PK\x03\x04truncated")
            linker_paths = LinkerPaths.from_directory(dir_name)
            with self.assertRaises(IOError):
                load_approximate_nearest_neighbours_index(linker_paths)

