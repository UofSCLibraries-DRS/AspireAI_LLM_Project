import os
from dotenv import load_dotenv, find_dotenv

from fine_tuning.utils.parse_pipeline import build_pipeline
from fine_tuning.inference import batched_inference


PIPELINE_PATH = "/home/john/Research/library/AspireAI_LLM_Project/fine_tuning/config/pipeline_2.json"


def main():
    load_dotenv(find_dotenv(".env.local"))

    pipeline = build_pipeline(
        pipeline_path=PIPELINE_PATH,
    )

    print(pipeline.inference_jobs[0])

    for train_step in pipeline.train_steps:
        train_step.train()

    res = batched_inference(pipeline.inference_jobs)

    print(res[0])


if __name__ == "__main__":
    main()
