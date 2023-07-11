import logging

from brain_ai.memory import Memory
from brain_ai.model_zoo.tabular_data_ai.execution import TabularAIExecutor
from brain_ai.model_zoo.text_data_ai.execution import SentimentDataExecutor
from brain_ai.utilities.data_handling import DataHandler


class ModelZoo:
    def __init__(self, configuration=None):
        self.configuration = configuration

    def base_models_train_and_test(self):
        for dataset in self.configuration['datasets']:
            for data_type, data in dataset.items():
                if data_type == 'Tabular_data':
                    print("reached here")

                    tabular_data = DataHandler(data['path']).dataframe()
                    print("it failed here", tabular_data)
                    target = tabular_data.pop(data['target'])
                    print(f"tabular_data: {tabular_data}, target: {target}")
                    return TabularAIExecutor(tabular_data, target, data['test_size']).execute_all_models()
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
