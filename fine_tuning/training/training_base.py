class AbstractTrainer:
    def __init__(
        self,
        start_model: str,
        output_dir: str,
        data: str,
        config: str,
    ):
        self.start_model = start_model
        self.output_dir = output_dir
        self.data = data
        self.config = config

    def train(self):
        pass
