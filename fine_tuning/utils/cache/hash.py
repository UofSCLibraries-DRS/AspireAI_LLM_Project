from typing import List
import hashlib

from fine_tuning.utils.cache.encoder import base62_encode


def list_hash(list: List[str]) -> str:
    return base62_encode(
        int.from_bytes(
            hashlib.sha256(("".join(list)).encode("utf-8")).digest(),
            byteorder="big",
        )
    )
