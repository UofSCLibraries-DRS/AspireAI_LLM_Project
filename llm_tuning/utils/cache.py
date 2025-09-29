import json
import os


class DummyCache:
    def __init__(self, *args, **kwargs):
        pass

    def dump(self):
        pass

    def load(self):
        pass


class TrainingCache:
    def __init__(self, path):
        self.cache = {}
        self.path = path
        self.load()

    def dump(self):
        with open(self.path, "w") as f:
            json.dump(self.cache, f, indent=2)

    def load(self):
        """Load cache from a JSON file."""
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    cache = json.load(f)
                print(f"Loaded {len(cache)} entries into cache from {self.path}.")
            except Exception as e:
                print(f"Error loading cache from {self.path}: {e}")
