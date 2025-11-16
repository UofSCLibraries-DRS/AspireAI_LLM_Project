import os
from dotenv import load_dotenv, find_dotenv

# from fine_tuning.utils.parse_pipeline import build_pipeline
# from fine_tuning.inference import batched_inference
from fine_tuning.trainers import LoRAUnsupervisedTrainer


PIPELINE_PATH = (
    "/home/john/Research/library/AspireAI_LLM_Project/fine_tuning/config/lora_test.json"
)


def main():
    # load_dotenv(find_dotenv(".env.local"))

    # pipeline = build_pipeline(
    #     pipeline_path=PIPELINE_PATH,
    # )

    # for train_step in pipeline.train_steps:
    #     train_step.train()

    print("Here 1")

    trainer = LoRAUnsupervisedTrainer(
        start_model="/work/jaaydin/models/gemma-3-270m",
        output_dir="/work/jaaydin/models/test_gemma",
        data="/work/jaaydin/raw/100_transcript.csv",
        config="/work/jaaydin/AspireAI_LLM_Project/fine_tuning/config/ft/training/lora_unsupervised_D_mcc_h2.json",
    )

    print("Here 2")

    trainer.train()

    # batched_inference(pipeline.inference_jobs)


if __name__ == "__main__":
    main()
