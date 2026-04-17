from typing import Dict, List, Tuple, Set
from collections import defaultdict
import copy


class PerClassScorer:
    def __init__(self):
        # These will hold per label span counts.
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)

    def __call__(
        self,
        predicted_spans: List[Tuple[int, int, str]],
        gold_spans: List[Tuple[int, int, str]],
    ) -> None:
        gold_spans = copy.copy(gold_spans)
        predicted_spans = copy.copy(predicted_spans)
        untyped_gold_spans = {(x[0], x[1]) for x in gold_spans}
        untyped_predicted_spans = {(x[0], x[1]) for x in predicted_spans}

        for untyped_span, span in zip(untyped_predicted_spans, predicted_spans):
            if span in gold_spans:
                self._true_positives[span[2]] += 1
                gold_spans.remove(span)
            else:
                self._false_positives[span[2]] += 1

            if untyped_span in untyped_gold_spans:
                self._true_positives["untyped"] += 1
                untyped_gold_spans.remove(untyped_span)
            else:
                self._false_positives["untyped"] += 1
        # These spans weren't predicted.
        for span in gold_spans:
            self._false_negatives[span[2]] += 1
        for untyped_span in untyped_gold_spans:
            self._false_negatives["untyped"] += 1

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A Dict per label containing following the span based metrics:
        precision : float
        recall : float
        f1-measure : float
        Additionally, an ``overall`` key is included, which provides the precision,
        recall and f1-measure for all spans.
        """
        all_tags: Set[str] = set()
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
        all_metrics = {}
        for tag in all_tags:
            precision, recall, f1_measure = self._compute_metrics(
                self._true_positives[tag],
                self._false_positives[tag],
                self._false_negatives[tag],
            )
            precision_key = "precision" + "-" + tag
            recall_key = "recall" + "-" + tag
            f1_key = "f1-measure" + "-" + tag
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure

        # Compute the precision, recall and f1 for all spans jointly.
        sum_true_positives = sum(
            v for k, v in self._true_positives.items() if k != "untyped"
        )
        sum_false_positives = sum(
            v for k, v in self._false_positives.items() if k != "untyped"
        )
        sum_false_negatives = sum(
            v for k, v in self._false_negatives.items() if k != "untyped"
        )
        precision, recall, f1_measure = self._compute_metrics(
            sum_true_positives, sum_false_positives, sum_false_negatives
        )
        all_metrics["precision-overall"] = precision
        all_metrics["recall-overall"] = recall
        all_metrics["f1-measure-overall"] = f1_measure
        if reset:
            self.reset()
        return all_metrics

    @staticmethod
    def _compute_metrics(
        true_positives: int, false_positives: int, false_negatives: int
    ):
        precision = float(true_positives) / float(
            true_positives + false_positives + 1e-13
        )
        recall = float(true_positives) / float(true_positives + false_negatives + 1e-13)
        f1_measure = 2.0 * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure

    def reset(self):
        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)
