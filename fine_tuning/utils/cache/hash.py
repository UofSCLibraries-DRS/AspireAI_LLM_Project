import hashlib
import json
from typing import Any, Sequence

from fine_tuning.utils.cache.encoder import base62_encode


def list_hash(values: Sequence[Any]) -> str:
    """Return a stable hash for an ordered sequence of JSON-compatible values."""
    serialized_values = json.dumps(
        values,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return base62_encode(
        int.from_bytes(
            hashlib.sha256(serialized_values.encode("utf-8")).digest(),
            byteorder="big",
        )
    )
