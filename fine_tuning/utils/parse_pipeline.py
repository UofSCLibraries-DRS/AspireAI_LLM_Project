import csv
import json
import os
from dataclasses import dataclass
from typing import Any

from fine_tuning.inference import InferenceJob
from fine_tuning.trainers import AbstractTrainer
from fine_tuning.utils.cache.hash import list_hash
from fine_tuning.utils.environment import get_env_or_raise
from fine_tuning.utils.validation import (
    validate_pipeline_json,
    validate_training_step_json,
)

# TODO: Ensure model / data folders exist / correct format (Maybe implement on trainer class)
#           Additionally, track model / data folders that will exist


@dataclass
class MasterPipeline:
    train_steps: list[AbstractTrainer]
    inference_jobs: list[InferenceJob]
    evaluation_steps: list[str]


def build_pipeline(
    pipeline_path: str,
) -> MasterPipeline:
    """
    `pipeline_path` - Relative path to pipeline json file (i.e. "config/pipeline.json")
    """
    MODEL_DUMP = get_env_or_raise("MODEL_DUMP")
    MODEL_FOLDER = get_env_or_raise("MODEL_FOLDER")
    DATA_FOLDER = get_env_or_raise("DATA_FOLDER")
    CONFIG_FOLDER = get_env_or_raise("CONFIG_FOLDER")
    PROMPT_FOLDER = get_env_or_raise("PROMPT_FOLDER")

    with open(pipeline_path, "r") as f:
        pipelines: list[dict] = json.load(f)

    validate_pipeline_json(pipelines)

    master_pipeline = MasterPipeline(
        train_steps=[],
        inference_jobs=[],
        evaluation_steps=[],
    )

    def load_training_step(training_step: Any) -> dict:
        """Return an inline step, loading legacy path-based steps when needed."""
        if isinstance(training_step, dict):
            return training_step

        with open(os.path.join(CONFIG_FOLDER, training_step), "r") as f:
            return json.load(f)

    # Normalize steps once so hashing and trainer construction use the same data.
    normalized_models = []
    for pipeline in pipelines:
        model = pipeline["model"]
        normalized_steps = [
            load_training_step(training_step)
            for training_step in model.get("train_steps", [])
        ]
        for training_step in normalized_steps:
            validate_training_step_json(training_step)
        normalized_models.append((model, normalized_steps))

    hash_master = {}

    # Initial loop:
    #   Creates hashes of all final output models (models that are given a name by the user)
    #   Prevents duplicate training steps.
    print("Gathering output models ...")
    for model, training_steps in normalized_models:
        # Parse model training
        # Get start and output and ensure they exist
        start = os.path.join(MODEL_FOLDER, model.get("start"))
        output = os.path.join(MODEL_FOLDER, model.get("output"))
        assert os.path.exists(start), (
            "error initializing pipeline: starting model does not exist"
        )
        os.makedirs(output, exist_ok=True)

        # Create a hash of the model being trained
        model_hash = list_hash([start, *training_steps])

        if model_hash in hash_master:
            raise ValueError("duplicate training pipelines")
        hash_master[model_hash] = output

    # Track seen models to avoid doubled training steps
    print("Building pipelines ...")
    seen_models = set()
    for pipeline, (model, training_steps) in zip(pipelines, normalized_models):
        ##########################
        ## Parse model training ##
        ##########################
        # Get only the start path (The output is already saved in the hash table)
        start = os.path.join(MODEL_FOLDER, model.get("start"))

        prev_model = start
        state = [start]
        trace = [start]  # Human-readable trace written by the trainer

        for training_step in training_steps:
            state.append(training_step)
            trace.append(json.dumps(training_step, sort_keys=True))

            state_hash = list_hash(state)

            named_output = hash_master.get(state_hash, None)
            if named_output:  # If there is a named output for this series of steps
                out_dir = named_output
            else:  # If there is not a named output for this series of steps
                out_dir = os.path.join(MODEL_DUMP, state_hash)

            # Check if this model has been added to the training jobs
            if state_hash in seen_models:
                # We'll need this later
                prev_model = out_dir
                continue
            seen_models.add(state_hash)

            trainer = AbstractTrainer.subclass_by_name(
                subclass_name=training_step.get("trainer"),
                start_model=prev_model,
                output_dir=out_dir,
                data=os.path.join(DATA_FOLDER, training_step.get("data")),
                config=os.path.join(CONFIG_FOLDER, training_step.get("config")),
                model_trace=trace.copy(),
            )

            master_pipeline.train_steps.append(trainer)

            prev_model = out_dir

        #####################
        ## Parse inference ##
        #####################

        # TODO: Rename
        inference_list: dict[str, str] | None = pipeline.get("inference")

        for inf in inference_list:
            with open(os.path.join(CONFIG_FOLDER, inf["config"]), "r") as f:
                inf_cfg: dict = json.load(f)
            # Create inference jobs from data
            for _data in inf["data"]:
                with open(os.path.join(DATA_FOLDER, _data), newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        q: str = row["question"]
                        gt_short: str = row["answer_short"]
                        gt_ideal: str = row["answer_ideal"]
                        gt_short_agg: str = row.get("answer_short_agg", "")
                        gt_ideal_agg: str = row.get("answer_ideal_agg", "")
                        dataset: str = row["dataset"]
                        subset: str = row["subset"]

                        for prompt_format in inf["prompt_formats"]:
                            inf_job = InferenceJob(
                                **inf_cfg,
                                output_file=_data,
                                model=os.path.join(
                                    MODEL_FOLDER, pipeline["model"]["output"]
                                ),
                                prompt_template=os.path.join(
                                    PROMPT_FOLDER, prompt_format
                                ),
                                prompt=q,
                                ground_truth_short=gt_short,
                                ground_truth_ideal=gt_ideal,
                                ground_truth_short_agg=gt_short_agg,
                                ground_truth_ideal_agg=gt_ideal_agg,
                                dataset=dataset,
                                subset=subset,
                            )

                            master_pipeline.inference_jobs.append(inf_job)
    return master_pipeline
