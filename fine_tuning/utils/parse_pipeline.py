from dataclasses import dataclass
import json
import os
from typing import List

from fine_tuning.trainers import AbstractTrainer
from fine_tuning.utils.cache.hash import list_hash
from fine_tuning.utils.validation import (
    validate_pipeline_json,
    validate_training_step_json,
)


# TODO: Ensure model / data folders exist / correct format (Maybe implement on trainer class)
#           Additionally, track model / data folders that will exist


@dataclass
class MasterPipeline:
    train_steps: List[AbstractTrainer]
    inference_steps: List[str]
    evaluation_steps: List[str]


def build_pipeline(
    pipeline_path: str,
    model_folder: str,
    model_dump: str,
) -> MasterPipeline:
    """
    `pipeline_path` - Relative path to pipeline json file (i.e. "config/pipeline.json")
    """

    with open(pipeline_path, "r") as f:
        pipelines: list[dict] = json.load(f)

    validate_pipeline_json(pipelines)

    master_pipeline = MasterPipeline(
        train_steps=[],
        inference_steps=[],
        evaluation_steps=[],
    )

    hash_master = {}

    # Initial loop:
    #   Creates hashes of all final output models (models that are given a name by the user)
    #   Prevents duplicate training steps.
    for pipeline in pipelines:
        # Parse model training
        model: dict = pipeline.get("model")

        # Get start and output and ensure they exist
        start = os.path.join(model_folder, model.get("start"))
        output = os.path.join(model_folder, model.get("output"))
        assert os.path.exists(start), (
            "error initializing pipeline: starting model does not exist"
        )
        os.makedirs(output, exist_ok=True)

        # Create a hash of the model being trained
        model_hash = list_hash([start] + model.get("train_steps"))

        if model_hash in hash_master:
            raise ValueError("duplicate training pipelines")
        hash_master[model_hash] = output

    # Track seen models to avoid doubled training steps
    seen_models = set()
    for pipeline in pipelines:
        # Parse model training
        model = pipeline.get("model")

        # Get only the start path (The output is already saved in the hash table)
        start = os.path.join(model_folder, model.get("start"))

        prev_model = start
        trace = [start]  # Store a trace of initial model and training steps

        for training_step_path in model.get("train_steps"):
            trace += [training_step_path]

            state_hash = list_hash(trace)  # Get hash of model

            named_output = hash_master.get(state_hash, None)
            if named_output:  # If there is a named output for this series of steps
                out_dir = named_output
            else:  # If there is not a named output for this series of steps
                out_dir = os.path.join(model_dump, state_hash)

            # Check if this model has been added to the training jobs
            if state_hash in seen_models:
                # We'll need this later
                prev_model = out_dir
                continue
            seen_models.add(state_hash)

            with open(training_step_path, "r") as f:
                training_step: dict = json.load(f)

            # Validate training step json
            validate_training_step_json(training_step)

            trainer = AbstractTrainer.subclass_by_name(
                subclass_name=training_step.get("trainer"),
                start_model=prev_model,
                output_dir=out_dir,
                data=training_step.get("data"),
                config=training_step.get("config"),
                model_trace=trace.copy(),
            )

            master_pipeline.train_steps.append(trainer)

            prev_model = out_dir

    return master_pipeline

    # for pipeline in pipelines:
    #     model = pipeline.get("model", None)
    #     evals = pipeline.get("eval", None)

    #     assert model, "❌ Pipeline defined with no model information"
    #     assert "start" in model, "❌ Pipeline defined with no start model"

    #     starting_models.append(model["start"])

    #     for step in model.get("steps", []):
    #         assert len(step) == 2, (
    #             "❌ Fine-tuning step defined without a clear action and data pair"
    #         )
    #         ft_actions.append(step[0])
    #         ft_data.append(step[1])

    #     for eval_section in eval:
    #         eval_data.extend(evals.get(eval_section, []))

    #     prompts.extend(pipeline.get("prompts"))

    #     # TODO: Ensure prompt - model validity

    #     # TODO: Ensure data - ft validity

    #     # TODO: Ensure

    # # Ensure models exist
    # unique_models = list(set(starting_models))
    # for model in unique_models:
    #     model_cfg_path = f"config/models/{model}"
    #     with open(model_cfg_path, "r") as f:
    #         model_cfg = json.load(f)

    #     model_path = model_cfg.get("path", None)
    #     assert model_path, f"❌ Model defined with no path: {model_cfg_path}"

    #     if not Path(model_path).is_dir():
    #         raise FileNotFoundError(f"Model not found: {model_path}")

    # # TODO: Ensure fine-tuning actions

    # # Ensure fine-tuning data exists
    # unique_ft_data = list(set(ft_data))
    # for ft_data in unique_ft_data:
    #     ft_data_cfg_path = f"config/training_data/{ft_data}"
    #     with open(ft_data_cfg_path, "r") as f:
    #         ft_data_cfg = json.load(f)

    #     ft_data_path = ft_data_cfg.get("path", None)
    #     assert ft_data_path, (
    #         f"❌ Training data defined with no path: {ft_data_cfg_path}"
    #     )

    #     if not Path(ft_data_path).is_file():
    #         raise FileNotFoundError(f"Training data not found: {ft_data_path}")

    # # Ensure prompt files exist and are in the correct format
    # unique_prompts = list(set(prompts))
    # for prompt in unique_prompts:
    #     prompt_path = Path(prompt)
    #     if not prompt_path.is_file():
    #         raise FileNotFoundError(f"❌ Prompt file not found: {prompt_path}")

    #     with open(prompt_path, "r") as f:
    #         prompt_cfg = yaml.safe_load(f)

    #     template = prompt_cfg.get("template", None)
    #     assert template, (
    #         f"❌ Prompt file missing required field 'template': {prompt_path}"
    #     )

    # # Ensure eval data exists
    # unique_eval_data = list(set(eval_data))
    # for eval_data in unique_eval_data:
    #     eval_data_cfg_path = f"config/eval_data/{eval_data}"
    #     with open(eval_data_cfg_path, "r") as f:
    #         eval_data_cfg = json.load(f)

    #     eval_data_path = eval_data_cfg.get("path", None)
    #     assert eval_data_path, (
    #         f"❌ Evalutation data defined with no path: {eval_data_cfg_path}"
    #     )

    #     if not Path(eval_data_path).is_file():
    #         raise FileNotFoundError(f"Evalutation data not found: {eval_data_path}")
