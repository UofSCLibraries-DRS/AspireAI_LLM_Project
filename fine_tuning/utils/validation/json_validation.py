from jsonschema import validate

PIPELINE_SCHEMA = {
    "type": "array",  # Outer array
    "items": {
        "type": "object",  # Individual pipeline
        "properties": {
            "model": {  # Model object
                "type": "object",
                "properties": {
                    "start": {"type": "string"},
                    "train_steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "output": {"type": "string"},
                },
            },
            "inference": {  # Inference object
                "type": "object",
            },
            "evaluation": {  # Evaluation object
                "type": "object",
            },
        },
    },
    "minItems": 1,
}

TRAINING_STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "trainer": {"type": "string"},
        "data": {"type": "string"},
        "config": {"type": "string"},
    },
}


def validate_pipeline_json(pipeline):
    validate(pipeline, PIPELINE_SCHEMA)


def validate_training_step_json(training_step):
    validate(training_step, TRAINING_STEP_SCHEMA)
