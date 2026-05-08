import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_PROMPT_TEMPLATE_PATH = "configs/templates/prompt/p1.yaml"
DEFAULT_RETRIEVED_ITEM_TEMPLATE_PATH = "configs/templates/retrieved_items/c1.yaml"
DEFAULT_QUESTION_COLUMN = "question"


@dataclass(frozen=True)
class RagConfig:
    embedding_model: str
    top_k: int
    prompt_template_path: str = DEFAULT_PROMPT_TEMPLATE_PATH
    retrieved_item_template_path: str = DEFAULT_RETRIEVED_ITEM_TEMPLATE_PATH


@dataclass(frozen=True)
class EvalDataConfig:
    path: str
    question_column: str = DEFAULT_QUESTION_COLUMN
    ground_truth_columns: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ChatbotSpec:
    id: str
    backend: str
    config: dict[str, Any]


@dataclass(frozen=True)
class ExperimentConfig:
    out: str
    eval_data: EvalDataConfig
    data: str
    rag_config: RagConfig
    chatbots: list[ChatbotSpec]


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)

    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Experiment config not found: {config_path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed experiment JSON at {config_path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"Experiment config at {config_path} must be a JSON object")

    out = _required_string(config, "out", config_path)
    eval_data = config.get("eval_data")
    if not isinstance(eval_data, dict):
        raise ValueError(f"Missing required object `eval_data` in {config_path}")

    data = _required_string(config, "data", config_path)
    rag_config = config.get("RAG_config")
    if not isinstance(rag_config, dict):
        raise ValueError(f"Missing required object `RAG_config` in {config_path}")

    return ExperimentConfig(
        out=out,
        eval_data=_load_eval_data_config(eval_data, config_path),
        data=data,
        rag_config=_load_rag_config(rag_config, config_path),
        chatbots=_load_chatbots(config.get("chatbots"), config_path),
    )


def _load_eval_data_config(config: dict[str, Any], config_path: Path) -> EvalDataConfig:
    location = f"{config_path} `eval_data`"
    return EvalDataConfig(
        path=_required_string(config, "path", location),
        question_column=_optional_string(
            config,
            "question_column",
            DEFAULT_QUESTION_COLUMN,
            location,
        ),
        ground_truth_columns=_required_string_list(
            config,
            "ground_truth_columns",
            location,
        ),
    )


def _load_rag_config(config: dict[str, Any], config_path: Path) -> RagConfig:
    location = f"{config_path} `RAG_config`"
    return RagConfig(
        embedding_model=_required_string(config, "embedding_model", location),
        top_k=_required_positive_int(config, "top_k", location),
        prompt_template_path=_optional_string(
            config,
            "prompt_template_path",
            DEFAULT_PROMPT_TEMPLATE_PATH,
            location,
        ),
        retrieved_item_template_path=_optional_string(
            config,
            "retrieved_item_template_path",
            DEFAULT_RETRIEVED_ITEM_TEMPLATE_PATH,
            location,
        ),
    )


def _load_chatbots(value: Any, config_path: Path) -> list[ChatbotSpec]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Missing required non-empty list `chatbots` in {config_path}")

    chatbots = []
    for index, item in enumerate(value):
        location = f"{config_path} `chatbots[{index}]`"
        if not isinstance(item, dict):
            raise ValueError(f"{location} must be an object")

        chatbot_config = item.get("config", {})
        if not isinstance(chatbot_config, dict):
            raise ValueError(f"{location} `config` must be an object")

        chatbots.append(
            ChatbotSpec(
                id=_required_string(item, "id", location),
                backend=_required_string(item, "backend", location),
                config=chatbot_config,
            )
        )
    return chatbots


def _required_string(config: dict[str, Any], key: str, location: object) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing required string `{key}` in {location}")
    return value


def _optional_string(
    config: dict[str, Any],
    key: str,
    default: str,
    location: object,
) -> str:
    value = config.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{key}` in {location} must be a non-empty string")
    return value


def _required_positive_int(config: dict[str, Any], key: str, location: object) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"Missing required positive integer `{key}` in {location}")
    return value


def _required_string_list(
    config: dict[str, Any],
    key: str,
    location: object,
) -> list[str]:
    value = config.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Missing required list `{key}` in {location}")

    items = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"`{key}[{index}]` in {location} must be a non-empty string"
            )
        items.append(item)
    return items
