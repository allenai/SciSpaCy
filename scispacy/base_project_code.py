from typing import Optional, Callable, Iterable, Iterator
from pathlib import Path

import random
import itertools
import spacy
import warnings
from spacy.training import Corpus, Example
from spacy.language import Language

from scispacy.custom_tokenizer import combined_rule_tokenizer
from scispacy.data_util import read_full_med_mentions, read_ner_from_tsv


def iter_sample(iterable: Iterable, sample_percent: float) -> Iterator:
    for item in iterable:
        if len(item.reference) == 0:
            continue
        coin_flip = random.uniform(0, 1)
        if coin_flip < sample_percent:
            yield item


@spacy.registry.callbacks("replace_tokenizer")
def replace_tokenizer_callback() -> Callable[[Language], Language]:
    def replace_tokenizer(nlp: Language) -> Language:
        nlp.tokenizer = combined_rule_tokenizer(nlp)
        return nlp

    return replace_tokenizer


@spacy.registry.readers("parser_tagger_data")
def parser_tagger_data(
    path: Path,
    mixin_data_path: Optional[Path],
    mixin_data_percent: float,
    gold_preproc: bool,
    max_length: int = 0,
    limit: int = 0,
    augmenter: Optional[Callable] = None,
    seed: int = 0,
) -> Callable[[Language], Iterator[Example]]:
    random.seed(seed)
    main_corpus = Corpus(
        path,
        gold_preproc=gold_preproc,
        max_length=max_length,
        limit=limit,
        augmenter=augmenter,
    )
    if mixin_data_path is not None:
        mixin_corpus = Corpus(
            mixin_data_path,
            gold_preproc=gold_preproc,
            max_length=max_length,
            limit=limit,
            augmenter=augmenter,
        )

    def mixed_corpus(nlp: Language) -> Iterator[Example]:
        if mixin_data_path is not None:
            main_examples = main_corpus(nlp)
            mixin_examples = iter_sample(mixin_corpus(nlp), mixin_data_percent)
            return itertools.chain(main_examples, mixin_examples)
        else:
            return main_corpus(nlp)

    return mixed_corpus


@spacy.registry.readers("med_mentions_reader")
def med_mentions_reader(
    directory_path: str, split: str
) -> Callable[[Language], Iterator[Example]]:
    train, dev, test = read_full_med_mentions(
        directory_path, label_mapping=None, span_only=True, spacy_format=True
    )

    def corpus(nlp: Language) -> Iterator[Example]:
        if split == "train":
            original_examples = train
        elif split == "dev":
            original_examples = dev
        elif split == "test":
            original_examples = test
        else:
            raise Exception(f"Unexpected split {split}")

        for original_example in original_examples:
            doc = nlp.make_doc(original_example[0])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                spacy_example = Example.from_dict(doc, original_example[1])
            yield spacy_example

    return corpus


@spacy.registry.readers("specialized_ner_reader")
def specialized_ner_reader(file_path: str) -> Callable[[Language], Iterator[Example]]:
    original_examples = read_ner_from_tsv(file_path)

    def corpus(nlp: Language) -> Iterator[Example]:
        for original_example in original_examples:
            doc = nlp.make_doc(original_example[0])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                spacy_example = Example.from_dict(doc, original_example[1])
            yield spacy_example

    return corpus
