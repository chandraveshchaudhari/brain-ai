import logging

from brain_ai.memory import Memory
from brain_ai.model_zoo.tabular_data_ai.execution import TabularAIExecutor
from brain_ai.model_zoo.text_data_ai.execution import SentimentDataExecutor


class ModelZoo:
    def __init__(self, memory_directory_path=None):
        self.memory = Memory(memory_directory_path)

    def base_models_train_and_test(self):
        for dataset in self.memory.configuration['datasets']:
            for data_type, data in dataset.items():
                if data_type == 'Tabular_data':
                    TabularAIExecutor(data['path'], data['target'], data['test_size']).execute_all_models()
                elif data_type == 'Text_data':
                    logging.info("using already trained")
                    # testing the already trained models
                    SentimentDataExecutor().execute_all_models(data)
                elif data_type == 'Time_series_data':
                    # todo: implement time series data
                    pass
                elif data_type == 'Image_data':
                    # todo: implement image data
                    pass
                else:
                    raise ValueError(f"Invalid data type {data_type}")

    def save_models_results(self):
        # use the memory to save the results by executing test datasets.
        pass

    def inference(self):
        pass
