from fine_tuning.trainers import AbstractTrainer, FullUnsupervisedTrainer
from fine_tuning.utils.parse_pipeline import build_pipeline


PIPELINE_PATH = (
    "/home/john/Research/library/AspireAI_LLM_Project/fine_tuning/config/loss_test.json"
)


def main():
    pipeline = build_pipeline(PIPELINE_PATH)

    for train_step in pipeline.train_steps:
        train_step.train()


if __name__ == "__main__":
    main()
