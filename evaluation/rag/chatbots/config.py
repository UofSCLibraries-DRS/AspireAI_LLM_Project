from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml


ConfigT = TypeVar("ConfigT")


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file)

    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping at the top level: {path}")
    return data


def config_from_mapping(
    config_cls: type[ConfigT],
    data: Mapping[str, Any],
    aliases: Mapping[str, str] | None = None,
) -> ConfigT:
    normalized = dict(data)
    for alias, canonical_name in (aliases or {}).items():
        if alias in normalized:
            if canonical_name not in normalized:
                normalized[canonical_name] = normalized[alias]
            del normalized[alias]

    field_names = {field.name for field in fields(config_cls)}
    unknown_fields = normalized.keys() - field_names
    if unknown_fields:
        raise ValueError(
            f"Unknown config fields for {config_cls.__name__}: {sorted(unknown_fields)}"
        )

    missing_fields = {
        field.name
        for field in fields(config_cls)
        if field.default is MISSING and field.default_factory is MISSING
    } - normalized.keys()
    if missing_fields:
        raise ValueError(
            f"Missing required config fields for {config_cls.__name__}: "
            f"{sorted(missing_fields)}"
        )

    return config_cls(**normalized)


def config_from_mapping_or_yaml(
    config_cls: type[ConfigT],
    config: Mapping[str, Any] | str | Path,
    aliases: Mapping[str, str] | None = None,
) -> ConfigT:
    if isinstance(config, str | Path):
        data = load_yaml_mapping(config)
    else:
        data = config

    return config_from_mapping(
        config_cls=config_cls,
        data=data,
        aliases=aliases,
    )
