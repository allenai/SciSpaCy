import json

import tqdm
from spacy.language import Language

from scispacy.per_class_scorer import PerClassScorer
from typing import Optional, Dict, Any


def evaluate_ner(
    nlp: Language, eval_data, dump_path: Optional[str] = None, verbose: bool = False
) -> Dict[str, Any]:
    scorer = PerClassScorer()
    print("Evaluating %d rows" % len(eval_data))
    for i, (text, gold_spans) in enumerate(tqdm.tqdm(eval_data)):
        # parse dev data with trained model
        doc = nlp(text)
        predicted_spans = [
            (ent.start_char, ent.end_char, ent.label_) for ent in doc.ents
        ]
        scorer(predicted_spans, gold_spans["entities"])

        if i % 1000 == 0 and i > 0:
            for name, metric in scorer.get_metric().items():
                print(f"{name}: {metric}")

    metrics = scorer.get_metric()
    if dump_path is not None:
        json.dump(metrics, open(dump_path, "a+"))
    for name, metric in metrics.items():
        if "overall" in name or "untyped" in name or verbose:
            print(f"{name}: \t\t {metric}")

    return metrics
