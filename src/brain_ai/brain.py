from brain_ai.memory import Memory


class Brain:
    def __init__(self, memory_directory_path=None, *args, **kwargs):
        self.memory = Memory(memory_directory_path)
        self.args = args
        self.kwargs = kwargs

    def generate_brain_configuration_file(self, output_configuration_file_path=None):
        config_file_path = self.memory.configuration_path if output_configuration_file_path is None else output_configuration_file_path
        self.memory.generate_configuration_file(config_file_path)








class ModelSelector:
    def __init__(self):
        self.dataset_list = []  # after separating the merged datasets
        self.model_list = []
        # check memory for configurations

    def select_model(self):
        pass

    def scale_dataset(self):
        pass

    def train_models(self):
        pass

    def inference(self):
        pass
