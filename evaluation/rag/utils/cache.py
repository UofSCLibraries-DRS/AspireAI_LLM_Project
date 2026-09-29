import hashlib
import json
from pathlib import Path


DEFAULT_CACHE_DIR = Path(".cache")


def get_embedding_cache_path(
    data: str,
    embedding_model: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Path:
    cache_key = {
        "data": data,
        "embedding_model": embedding_model,
    }
    cache_key_json = json.dumps(
        cache_key,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(cache_key_json.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.csv"
