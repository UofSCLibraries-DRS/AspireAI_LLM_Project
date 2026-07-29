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
                        "items": {
                            "oneOf": [
                                {"type": "string"},
                                {
                                    "type": "object",
                                    "properties": {
                                        "trainer": {"type": "string"},
                                        "data": {"type": "string"},
                                        "config": {"type": "string"},
                                    },
                                    "required": ["trainer", "data", "config"],
                                },
                            ]
                        },
                    },
                    "output": {"type": "string"},
                },
            },
            "inference": {  # Inference object
                "type": "array",
            },
            "evaluation": {  # Evaluation object
                "type": "object",
            },
        },
        "required": ["model"],
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
    "required": ["trainer", "data", "config"],
}


def validate_pipeline_json(pipeline):
    validate(pipeline, PIPELINE_SCHEMA)


def validate_training_step_json(training_step):
    validate(training_step, TRAINING_STEP_SCHEMA)
