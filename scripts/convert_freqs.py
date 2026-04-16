import argparse
import math
import json
from ast import literal_eval
from tqdm import tqdm
from preshed.counter import PreshCounter
from spacy.util import ensure_path
from scispacy.file_cache import cached_path


def read_freqs(freqs_loc, max_length=100, min_doc_freq=5, min_freq=50):
    print("Counting frequencies...")
    counts = PreshCounter()
    total = 0
    with freqs_loc.open() as f:
        for i, line in tqdm(enumerate(f)):
            freq, doc_freq, key = line.rstrip().split("\t", 2)
            freq = int(freq)
            counts.inc(i + 1, freq)
            total += freq
    counts.smooth()
    log_total = math.log(total)
    probs = {}
    with freqs_loc.open() as f:
        for line in tqdm(f):
            freq, doc_freq, key = line.rstrip().split("\t", 2)
            doc_freq = int(doc_freq)
            freq = int(freq)
            if doc_freq >= min_doc_freq and freq >= min_freq and len(key) < max_length:
                try:
                    word = literal_eval(key)
                except SyntaxError:
                    # Take odd strings literally.
                    word = literal_eval("'%s'" % key)
                smooth_count = counts.smoother(int(freq))
                probs[word] = math.log(smooth_count) - log_total
    oov_prob = math.log(counts.smoother(0)) - log_total
    return probs, oov_prob


def main(input_path: str, output_path: str, min_word_frequency: int):
    if input_path is not None:
        input_path = cached_path(input_path)
        input_path = ensure_path(input_path)

    probs, oov_prob = (
        read_freqs(input_path, min_freq=min_word_frequency)
        if input_path is not None
        else ({}, -20)
    )

    with open(output_path, "w") as _jsonl_file:
        _jsonl_file.write(
            json.dumps({"lang": "en", "settings": {"oov_prob": oov_prob}})
        )
        _jsonl_file.write("\n")

        for word, prob in probs.items():
            _jsonl_file.write(json.dumps({"orth": word, "prob": prob}))
            _jsonl_file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", type=str, default=None, help="Path to the freqs file"
    )
    parser.add_argument(
        "--output_path", type=str, help="Output path for the jsonl file"
    )
    parser.add_argument(
        "--min_word_frequency",
        type=int,
        default=50,
        help="Minimum word frequency for inclusion",
    )

    args = parser.parse_args()
    main(args.input_path, args.output_path, args.min_word_frequency)