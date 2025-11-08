class MissingEnvironmentVariable(Exception):
    def __init__(self, var_name: str, message: str | None = None) -> None:
        if message is None:
            message = f"Required environment variable '{var_name}' is missing."
        super().__init__(message)
        self.var_name = var_name
