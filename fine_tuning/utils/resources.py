import json
from pathlib import Path

import yaml


def ensure_pipeline(pipeline_path, base_dir):
    """
    `pipeline_path` - Relative path to pipeline json file (i.e. "config/pipeline.json")

    `base_dir` - Absolute path to the directory where models, datasets, and outputs are saved
    """

    with open(pipeline_path, "r") as f:
        pipelines = json.load(f)

    # Ensure all starting models are available

    starting_models = []
    ft_actions = []
    ft_data = []
    eval_data = []
    prompts = []

    for pipeline in pipelines:
        model = pipeline.get("model", None)
        evals = pipeline.get("eval", None)

        assert model, "❌ Pipeline defined with no model information"
        assert "start" in model, "❌ Pipeline defined with no start model"

        starting_models.append(model["start"])

        for step in model.get("steps", []):
            assert len(step) == 2, (
                "❌ Fine-tuning step defined without a clear action and data pair"
            )
            ft_actions.append(step[0])
            ft_data.append(step[1])

        for eval_section in eval:
            eval_data.extend(evals.get(eval_section, []))

        prompts.extend(pipeline.get("prompts"))

        # TODO: Ensure prompt - model validity

        # TODO: Ensure data - ft validity

        # TODO: Ensure

    # Ensure models exist
    unique_models = list(set(starting_models))
    for model in unique_models:
        model_cfg_path = f"config/models/{model}"
        with open(model_cfg_path, "r") as f:
            model_cfg = json.load(f)

        model_path = model_cfg.get("path", None)
        assert model_path, f"❌ Model defined with no path: {model_cfg_path}"

        if not Path(model_path).is_dir():
            raise FileNotFoundError(f"Model not found: {model_path}")

    # TODO: Ensure fine-tuning actions

    # Ensure fine-tuning data exists
    unique_ft_data = list(set(ft_data))
    for ft_data in unique_ft_data:
        ft_data_cfg_path = f"config/training_data/{ft_data}"
        with open(ft_data_cfg_path, "r") as f:
            ft_data_cfg = json.load(f)

        ft_data_path = ft_data_cfg.get("path", None)
        assert ft_data_path, (
            f"❌ Training data defined with no path: {ft_data_cfg_path}"
        )

        if not Path(ft_data_path).is_file():
            raise FileNotFoundError(f"Training data not found: {ft_data_path}")

    # Ensure prompt files exist and are in the correct format
    unique_prompts = list(set(prompts))
    for prompt in unique_prompts:
        prompt_path = Path(prompt)
        if not prompt_path.is_file():
            raise FileNotFoundError(f"❌ Prompt file not found: {prompt_path}")

        with open(prompt_path, "r") as f:
            prompt_cfg = yaml.safe_load(f)

        template = prompt_cfg.get("template", None)
        assert template, (
            f"❌ Prompt file missing required field 'template': {prompt_path}"
        )

    # Ensure eval data exists
    unique_eval_data = list(set(eval_data))
    for eval_data in unique_eval_data:
        eval_data_cfg_path = f"config/eval_data/{eval_data}"
        with open(eval_data_cfg_path, "r") as f:
            eval_data_cfg = json.load(f)

        eval_data_path = eval_data_cfg.get("path", None)
        assert eval_data_path, (
            f"❌ Evalutation data defined with no path: {eval_data_cfg_path}"
        )

        if not Path(eval_data_path).is_file():
            raise FileNotFoundError(f"Evalutation data not found: {eval_data_path}")
