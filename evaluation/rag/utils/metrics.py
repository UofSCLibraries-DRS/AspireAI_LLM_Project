from typing import Any, ClassVar, Iterable

import numpy as np
from gaico.metrics.base import BaseMetric


class SentenceTransformerCosine(BaseMetric):
    _model: ClassVar[Any | None] = None

    def __init__(self, seed: int | None = None, **kwargs: Any) -> None:
        super().__init__(seed=seed, **kwargs)
        if self.__class__._model is None:
            from sentence_transformers import SentenceTransformer

            self.__class__._model = SentenceTransformer("all-MiniLM-L6-v2")

    def _single_calculate(
        self,
        generated_item: str,
        reference_item: str,
        **kwargs: Any,
    ) -> float:
        return self._batch_calculate([generated_item], [reference_item], **kwargs)[0]

    def _batch_calculate(
        self,
        generated_items: Iterable,
        reference_items: Iterable,
        **kwargs: Any,
    ) -> list[float]:
        generated_embeddings = np.asarray(
            self._model.encode(list(map(str, generated_items)), normalize_embeddings=True)
        )
        reference_embeddings = np.asarray(
            self._model.encode(list(map(str, reference_items)), normalize_embeddings=True)
        )
        return np.clip(
            np.sum(generated_embeddings * reference_embeddings, axis=1),
            -1.0,
            1.0,
        ).tolist()


class CachedBERTScore(BaseMetric):
    _scorer: ClassVar[Any | None] = None

    def __init__(self, seed: int | None = None, **kwargs: Any) -> None:
        super().__init__(seed=seed, **kwargs)
        if self.__class__._scorer is None:
            from bert_score import BERTScorer

            self.__class__._scorer = BERTScorer(
                model_type="bert-base-uncased",
                num_layers=8,
                batch_size=64,
            )

    def _single_calculate(
        self,
        generated_item: str,
        reference_item: str,
        **kwargs: Any,
    ) -> dict[str, float]:
        return self._batch_calculate([generated_item], [reference_item], **kwargs)[0]

    def _batch_calculate(
        self,
        generated_items: Iterable,
        reference_items: Iterable,
        **kwargs: Any,
    ) -> list[dict[str, float]]:
        precision, recall, f1 = self._scorer.score(
            list(map(str, generated_items)),
            list(map(str, reference_items)),
        )
        return [
            {
                "precision": float(item_precision),
                "recall": float(item_recall),
                "f1": float(item_f1),
            }
            for item_precision, item_recall, item_f1 in zip(
                precision,
                recall,
                f1,
                strict=True,
            )
        ]
