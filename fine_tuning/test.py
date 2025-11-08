import os
from dotenv import load_dotenv, find_dotenv

from fine_tuning.utils.exceptions import MissingEnvironmentVariable
from fine_tuning.trainers import AbstractTrainer, FullUnsupervisedTrainer
from fine_tuning.utils.parse_pipeline import build_pipeline


PIPELINE_PATH = "/home/john/Research/library/AspireAI_LLM_Project/fine_tuning/config/gemma_pipeline.json"


def main():
    load_dotenv(find_dotenv(".env.local"))

    MODEL_DUMP = os.getenv("MODEL_DUMP")
    MODEL_FOLDER = os.getenv("MODEL_FOLDER")

    if MODEL_DUMP is None:
        raise MissingEnvironmentVariable("MODEL_DUMP")

    pipeline = build_pipeline(
        pipeline_path=PIPELINE_PATH,
        model_folder=MODEL_FOLDER,
        model_dump=MODEL_DUMP,
    )

    for train_step in pipeline.train_steps:
        train_step.train()


if __name__ == "__main__":
    main()
