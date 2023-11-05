import logging

from brain_automl.model_zoo.tabular_data_ai.AutoML import TabularAIExecutor
from brain_automl.model_zoo.text_data_ai.execution import SentimentDataExecutor
from brain_automl.utilities.data_handling import DataHandler


class ModelZoo:
    def __init__(self, configuration=None):
        self.configuration = configuration
        self.metric = dict()

    def base_models_train_and_test(self):
        for dataset in self.configuration['datasets']:
            for data_type, data in dataset.items():
                if data_type == 'Tabular_data':
                    tabular_data = DataHandler(data['path']).dataframe()
                    target_column_name = data['target']
                    print(f"tabular_data: {tabular_data}, target: {target_column_name}")

                    tracking_uri = self.configuration['tracking_uri'] if 'tracking_uri' in self.configuration else None
                    self.metric['Tabular_data'] = TabularAIExecutor(tabular_data, target_column_name, data['test_size'],
                                                                    tracking_uri).train_and_test()
                elif data_type == 'Sentiment_data':
                    logging.info("using already trained")
                    # testing the already trained models
                    # testing dataset
                    tabular_data = DataHandler(data['path']).dataframe()
                    target_column_name = data['target']
                    if 'memory_check' in data and data['memory_check']:
                        test_result = data['path'].split('/')[-1].split('.')[0] + '_test_result.csv'
                        self.metric['Sentiment_data'] = DataHandler(test_result).dataframe()
                    else:
                        self.metric['Sentiment_data'] = SentimentDataExecutor(tabular_data,
                                                                              target_column_name).add_result_column()
                elif data_type == 'Time_series_data':
                    # todo: implement time series data
                    pass
                elif data_type == 'Image_data':
                    # todo: implement image data
                    pass
                else:
                    raise ValueError(f"Invalid data type {data_type}")
        return self.metric

    def save_models_results(self):
        # use the memory to save the results by executing test datasets.
        pass

    def inference(self):

        pass
