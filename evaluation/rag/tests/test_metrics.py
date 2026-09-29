import unittest

import numpy as np

from utils.metrics import CachedBERTScore, SentenceTransformerCosine


class MetricTests(unittest.TestCase):
    def tearDown(self) -> None:
        SentenceTransformerCosine._model = None
        CachedBERTScore._scorer = None

    def test_sentence_transformer_cosine_uses_normalized_embeddings(self) -> None:
        model = FakeSentenceTransformer(
            [
                np.array([[1.0, 0.0], [0.0, 1.0]]),
                np.array([[1.0, 0.0], [1.0, 0.0]]),
            ]
        )
        SentenceTransformerCosine._model = model

        scores = SentenceTransformerCosine().calculate(
            ["same", "different"],
            ["same", "other"],
        )

        self.assertEqual(scores, [1.0, 0.0])
        self.assertEqual(
            model.calls,
            [
                (["same", "different"], True),
                (["same", "other"], True),
            ],
        )

    def test_bert_score_returns_precision_recall_and_f1(self) -> None:
        CachedBERTScore._scorer = FakeBERTScorer()

        scores = CachedBERTScore().calculate(["response"], ["reference"])

        self.assertEqual(
            scores,
            [{"precision": 0.7, "recall": 0.8, "f1": 0.75}],
        )


class FakeSentenceTransformer:
    def __init__(self, outputs: list[np.ndarray]) -> None:
        self.outputs = iter(outputs)
        self.calls: list[tuple[list[str], bool]] = []

    def encode(self, texts: list[str], normalize_embeddings: bool) -> np.ndarray:
        self.calls.append((texts, normalize_embeddings))
        return next(self.outputs)


class FakeBERTScorer:
    def score(
        self,
        generated: list[str],
        references: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.array([0.7]), np.array([0.8]), np.array([0.75])


if __name__ == "__main__":
    unittest.main()
