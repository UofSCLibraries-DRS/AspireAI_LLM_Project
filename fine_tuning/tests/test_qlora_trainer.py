import ast
import json
import unittest
from pathlib import Path


FINE_TUNING_DIR = Path(__file__).parents[1]
QLORA_TRAINER_PATH = FINE_TUNING_DIR / "trainers" / "qlora_unsupervised.py"
QLORA_CONFIG_PATH = FINE_TUNING_DIR / "config" / "fine-tuning" / "qlora_unsupervised_100.json"


class QLoRATrainerStructureTest(unittest.TestCase):
    def test_uses_nf4_kbit_preparation_and_saves_an_adapter(self):
        tree = ast.parse(QLORA_TRAINER_PATH.read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertTrue(
            {"BitsAndBytesConfig", "prepare_model_for_kbit_training"} <= names
        )

        trainer_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "QLoRAUnsupervisedTrainer"
        )
        train_method = next(
            node
            for node in trainer_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "_train"
        )
        self.assertIn(
            "resume_from_checkpoint", [argument.arg for argument in train_method.args.args]
        )

        calls = [
            node
            for node in ast.walk(train_method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "trainer"
            and node.func.attr == "train"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            [keyword.arg for keyword in calls[0].keywords], ["resume_from_checkpoint"]
        )

        attributes = [
            node.func.attr
            for node in ast.walk(train_method)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        self.assertIn("save_pretrained", attributes)
        self.assertNotIn("merge_and_unload", attributes)

    def test_example_config_is_valid_qlora_configuration(self):
        config = json.loads(QLORA_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(config["quantization_config"]["compute_dtype"], "bfloat16")
        self.assertTrue(
            config["quantization_config"]["bnb_4bit_use_double_quant"]
        )
        self.assertEqual(config["lora_config"]["task_type"], "CAUSAL_LM")


if __name__ == "__main__":
    unittest.main()
