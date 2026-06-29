#!/usr/bin/env python

from typing import List, Tuple
import os
import io
import sys
import tempfile
import shutil
from collections import Counter
from pathlib import Path
from multiprocessing import Pool

import plac

import spacy.util
from spacy.language import Language

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

from scispacy.custom_tokenizer import combined_rule_tokenizer

def count_frequencies(language_class: Language, input_path: Path):
    """
    Given a file containing single documents per line
    (for scispacy, these are Pubmed abstracts), split the text
    using a science specific tokenizer and compute word and
    document frequencies for all words.
    """
    print(f"Processing {input_path}.")
    tokenizer = combined_rule_tokenizer(language_class())
    counts = Counter()
    doc_counts = Counter()
    with open(input_path, "r") as input_file:
        for line in input_file:
            words = [t.text for t in tokenizer(line)]
            counts.update(words)
            doc_counts.update(set(words))

    return counts, doc_counts

def parallelize(func, iterator, n_jobs):
    pool = Pool(processes=n_jobs)
    counts = pool.starmap(func, iterator)
    return counts

def merge_counts(frequencies: List[Tuple[Counter, Counter]], output_path: str):
    """
    Merge a number of frequency counts generated from `count_frequencies`
    into a single file, written to `output_path`.
    """
    counts = Counter()
    doc_counts = Counter()
    for word_count, doc_count in frequencies:
        counts.update(word_count)
        doc_counts.update(doc_count)
    with io.open(output_path, 'w+', encoding='utf8') as file_:
        for word, count in counts.most_common():
            if not word.isspace():
                file_.write(f"{count}\t{doc_counts[word]}\t{repr(word)}\n")


@plac.annotations(
        raw_dir=("Location of input file list", "positional", None, Path),
        output_dir=("Location for output file", "positional", None, Path),
        n_jobs=("Number of workers", "option", "n", int))
def main(raw_dir: Path, output_dir: Path, n_jobs=2):

    language_class = spacy.util.get_lang_class("en")
    tasks = []
    freqs_dir = Path(tempfile.mkdtemp(prefix="scispacy_freqs"))
    for input_path in [os.path.join(raw_dir, filename)
                       for filename in os.listdir(raw_dir)]:
        input_path = Path(input_path.strip())
        if not input_path:
            continue
        tasks.append((language_class, input_path))

    if tasks:
        counts = parallelize(count_frequencies, tasks, n_jobs)

    print("Merge")
    merge_counts(counts, output_dir)
    shutil.rmtree(freqs_dir)

if __name__ == '__main__':
    plac.call(main)
