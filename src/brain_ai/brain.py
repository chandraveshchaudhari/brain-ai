from brain_ai.memory import Memory
from brain_ai.model_zoo.model_selector_and_trainer import ModelZoo


class Brain:
    def __init__(self, memory_directory_path=None, *args, **kwargs):
        self.memory = Memory(memory_directory_path)
        self.args = args
        self.kwargs = kwargs

    def generate_brain_configuration_file(self, output_configuration_file_path=None):
        config_file_path = self.memory.configuration_path if output_configuration_file_path is None else output_configuration_file_path
        self.memory.generate_configuration_file(config_file_path)

    def train(self):
        ModelZoo(self.memory).train()












