import logging

from brain_ai.memory import Memory
from brain_ai.model_zoo.model_selector_and_trainer import ModelZoo
from brain_ai.model_zoo.tabular_data_ai.execution import TabularAIExecutor


class Brain(Memory):
    def __init__(self, memory_directory_path=None, *args, **kwargs):
        super().__init__(memory_directory_path)
        self.args = args
        self.kwargs = kwargs

    def train(self):
        print("Training the models", self.configuration)
        logging.info("Training the models")
        metric = ModelZoo(self.configuration).base_models_train_and_test()
        # TabularAIExecutor(base_models_result_dataset, target).execute_all_models()
        return metric

    def inference(self):
        # get the saved models from the memory and use them to predict the new data.
        pass

