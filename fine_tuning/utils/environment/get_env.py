import os

from .exceptions import MissingEnvironmentVariable


def get_env_or_raise(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise MissingEnvironmentVariable(name)
    return value
