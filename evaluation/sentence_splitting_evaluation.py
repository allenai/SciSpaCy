import argparse
import os
import sys
import json

import spacy

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from scispacy.custom_sentence_segmenter import combined_rule_sentence_segmenter
from scispacy.custom_tokenizer import remove_new_lines, combined_rule_tokenizer

def evaluate_sentence_splitting(model_path: str,
                                data_directory: str,
                                rule_segmenter: bool = False,
                                custom_tokenizer: bool = False,
                                citation_data_path: str = None):

    model = spacy.load(model_path)
    if rule_segmenter:
        model.add_pipe(combined_rule_sentence_segmenter, first=True)
    if custom_tokenizer:
        model.tokenizer = combined_rule_tokenizer(model)

    total_correct = 0
    total = 0
    total_abstracts = 0
    perfect = 0
    for abstract_name in os.listdir(data_directory):

        with open(os.path.join(data_directory, abstract_name), "r") as abstract_file:
            abstract_sentences = [x.strip() for x in abstract_file]

        full_abstract = " ".join(abstract_sentences)

        doc = model(full_abstract)

        sentences = [x.text for x in doc.sents]

        correct = []
        for sentence in sentences:
            if sentence in abstract_sentences:
                correct.append(1)
            else:
                correct.append(0)

        total += len(correct)
        total_correct += sum(correct)
        perfect += all(correct)
        total_abstracts += 1

    print(f"Sentence splitting performance for {model_path} :\n")

    print(f"Sentence level accuracy: {total_correct} of {total}, {total_correct / total}. ")
    print(f"Abstract level accuracy: {perfect} of {total_abstracts}, {perfect / total_abstracts}. ")

    if citation_data_path is None:
        return

    skipped = 0
    citation_total = 0
    citation_correct = 0
    for line in open(citation_data_path, "r"):

        sentence = remove_new_lines(json.loads(line)["string"])

        # Skip sentence if it doesn't look roughly like a sentence,
        # or it is > 2 std deviations above the mean length.
        if not sentence[0].isupper() or sentence[-1] != "." or len(sentence) > 450:
            skipped += 1
            continue

        sentences = list(model(sentence).sents)

        if len(sentences) == 1:
            citation_correct += 1
        citation_total +=1
    print(f"Citation handling performance for {model_path}, skipped {skipped} examples :\n")
    print(f"Citation level accuracy: {citation_correct} of {citation_total}, {citation_correct / citation_total}. ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data',
        help="Path to the directory containing the raw data."
    )
    parser.add_argument(
        '--model_path',
        default=None,
        help="Path to the spacy model to load"
    )
    parser.add_argument(
        '--rule_segmenter',
        default=False,
        action="store_true",
        help="Whether to use the rule based segmenter"
    )
    parser.add_argument(
        '--custom_tokenizer',
        default=False,
        action="store_true",
        help="Whether to use the rule based segmenter"
    )
    parser.add_argument(
        '--citation_data',
        default=None,
        help="Path to the jsonl file containing the citation contexts."
    )

    args = parser.parse_args()
    evaluate_sentence_splitting(args.model_path, args.data, args.rule_segmenter, args.custom_tokenizer, args.citation_data)
