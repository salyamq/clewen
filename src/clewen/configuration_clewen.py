"""hugging Face configuration for a two-mode Clewen inference release"""

from transformers import PretrainedConfig


class ClewenConfig(PretrainedConfig):
    model_type = "clewen"

    def __init__(self, default_max_input_tokens=4096, source_models=None, modes=None, format_version=2, **kwargs):
        super().__init__(**kwargs)
        self.default_max_input_tokens = default_max_input_tokens
        self.source_models = source_models or {}
        self.modes = ["text", "decision"] if modes is None else modes
        self.format_version = format_version
        if type(default_max_input_tokens) is not int or default_max_input_tokens < 1:
            raise ValueError("default_max_input_tokens must be a positive integer")
        if self.modes != ["text", "decision"]:
            raise ValueError("Clewen supports explicit text and decision modes")
