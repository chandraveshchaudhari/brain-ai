from brain_ai.memory import Memory
from brain_ai.model_zoo.model_selector_and_trainer import ModelZoo
from brain_ai.model_zoo.tabular_data_ai.execution import TabularAIExecutor


class Brain:
    def __init__(self, memory_directory_path=None, *args, **kwargs):
        self.memory = Memory(memory_directory_path)
        self.args = args
        self.kwargs = kwargs

    def generate_brain_configuration_file(self, output_configuration_file_path=None):
        config_file_path = self.memory.configuration_path if output_configuration_file_path is None else output_configuration_file_path
        self.memory.generate_configuration_file(config_file_path)

    def train(self):
        base_models_result_dataset, target = ModelZoo(self.memory).base_models_train_and_test()
        TabularAIExecutor(base_models_result_dataset, target).execute_all_models()

    def inference(self):
        # get the saved models from the memory and use them to predict the new data.
        pass

