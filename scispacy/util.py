from packaging.version import Version
import spacy
import scipy
from spacy.language import Language
from spacy.tokens import Doc
from spacy.vocab import Vocab

from scispacy.custom_sentence_segmenter import pysbd_sentencizer
from scispacy.custom_tokenizer import combined_rule_tokenizer


def save_model(nlp: Language, output_path: str) -> None:
    nlp.to_disk(output_path)


def create_combined_rule_model() -> Language:
    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = combined_rule_tokenizer(nlp)
    nlp.add_pipe(pysbd_sentencizer, first=True)
    return nlp


def scipy_supports_sparse_float16() -> bool:
    # https://github.com/scipy/scipy/issues/7408
    return Version(scipy.__version__) < Version("1.11")


class WhitespaceTokenizer:
    """
    Spacy doesn't assume that text is tokenised. Sometimes this
    is annoying, like when you have gold data which is pre-tokenised,
    but Spacy's tokenisation doesn't match the gold. This can be used
    as follows:
    nlp = spacy.load("en_core_web_md")
    # hack to replace tokenizer with a whitespace tokenizer
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    ... use nlp("here is some text") as normal.
    """

    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab

    def __call__(self, text: str) -> Doc:
        words = text.split(" ")
        # All tokens 'own' a subsequent space character in
        # this tokenizer. This is a technicality and probably
        # not that interesting.
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)
