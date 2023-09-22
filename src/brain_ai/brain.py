import os

from brain_ai.explainability.exploratory_data_analysis import find_common_columns
from brain_ai.memory import Memory
from brain_ai.model_zoo.tabular_data_ai.AutoML import TabularAutoML
from brain_ai.model_zoo.text_data_ai.execution import SentimentDataExecutor
from brain_ai.utilities.data_handling import DataHandler
from brain_ai.utilities.dataset_merger import Merge
from brain_ai.utilities.log_handling import Logger


class Brain(Memory):
    def __init__(self, configuration_dict_or_path=None, project_name="BrainAutoML",
                 memory_directory_path=os.getcwd(), *args, **kwargs):

        super().__init__(memory_directory_path=memory_directory_path,
                         configuration_dict_or_path=configuration_dict_or_path,
                         project_name=project_name)
        self.project_name = project_name
        self.args = args
        self.kwargs = kwargs

        self.logger = Logger(log_project_name=project_name, log_directory_path=self.directories_created[1])
        self.logger.welcome_log(project_name)
        self.logger.info(f"{project_name} started:"
                         f"Memory directory path is {self.directories_created[0]}"
                         f"logs directory path is {self.directories_created[1]}"
                         f"saved models directory path is {self.directories_created[2]}"
                         f"predicted data directory path is {self.directories_created[3]}"
                         f"generated data directory path is {self.directories_created[4]}")

    def train_and_test_brain(self):
        self.training_and_testing_subpart_of_brain()
        self.merge_dataset()
        brain_object = TabularAutoML(self.configuration['Merged Dataset Path'],
                      'target', logger=self.logger,
                      tabular_directory=self.directories_created[0],
                      project_name=self.project_name).train_predict_save_metrics(clean_data=self.configuration['clean_data'])
        prediction_path = os.path.join(brain_object.saved_models_directory_path, 'performance_metrics.csv')
        self.configuration['prediction_path'] = prediction_path
        return brain_object.performance_metrics()

    def training_and_testing_subpart_of_brain(self):

        if 'datasets' in self.configuration:
            for dataset_type, dataset_info in self.configuration['datasets'].items():
                if dataset_type.endswith('Tabular_data'):
                    if 'prediction_dictionary_path' not in dataset_info:

                        if dataset_info['Generated_dataset'] is True:
                            continue
                        if 'split_data_by_column_name_and_value_dict' in dataset_info:
                            tabular_automl_object = TabularAutoML(dataset_info['path'], dataset_info['target'],
                                                                  split_data_by_column_name_and_value_dict=dataset_info[
                                                                      'split_data_by_column_name_and_value_dict'],
                                                                  logger=self.logger,
                                                                  tabular_directory=self.directories_created[0])
                            dataset_info['Directories Created'] = tabular_automl_object.directories_created
                            tabular_automl_object.train_predict_save_metrics(clean_data=dataset_info['clean_data'])

                        elif 'test_size' in dataset_info:
                            tabular_automl_object = TabularAutoML(dataset_info['path'], dataset_info['target'],
                                                                  test_size=dataset_info['test_size'],
                                                                  logger=self.logger,
                                                                  tabular_directory=self.directories_created[0])
                            dataset_info['Directories Created'] = tabular_automl_object.directories_created
                            tabular_automl_object.train_predict_save_metrics(clean_data=dataset_info['clean_data'])
                            tabular_automl_object.best_model_prediction_path()
                        else:
                            tabular_automl_object = TabularAutoML(dataset_info['path'], dataset_info['target'],
                                                                  logger=self.logger,
                                                                  tabular_directory=self.directories_created[0])
                            dataset_info['Directories Created'] = tabular_automl_object.directories_created
                            tabular_automl_object.train_predict_save_metrics(clean_data=dataset_info['clean_data'])
                            tabular_automl_object.best_model_prediction_path()
                elif dataset_type.endswith('Sentiment_data'):
                    if 'prediction_dictionary_path' not in dataset_info:
                        sentiment_data = SentimentDataExecutor(dataset_info['path'],
                                                               dataset_info['target']).add_result_column()
                        sentiment_data_path = os.path.join(self.directories_created[4], 'sentiment_data.csv')
                        dataset_info['y_pred'] = sentiment_data_path
                        DataHandler().write(sentiment_data_path, sentiment_data)
                else:
                    raise NotImplementedError(f"Unknown dataset type: {dataset_type}")

    def merge_dataset(self):
        # TODO: add the functionality to merge the dataset (too complicated to do it for unknown data types)

        if 'merged_dataset' not in self.configuration:
            list_of_dataset = []
            for dataset_type, dataset_info in self.configuration['datasets'].items():
                if dataset_type.endswith('Tabular_data'):
                    if 'Directories Created' in dataset_info:
                        generated_data_dictionary_path = os.path.join(dataset_info['Directories Created'][4],
                                                                      'generated_data_dictionary.json')
                        generated_data_dictionary = DataHandler(generated_data_dictionary_path).load()
                        x_test_df = DataHandler(generated_data_dictionary['x_test']).dataframe()
                        y_test_df = DataHandler(generated_data_dictionary['y_test']).dataframe()

                        prediction_dictionary_path = dataset_info['prediction_dictionary_path']
                        prediction_dictionary = DataHandler(prediction_dictionary_path).load()
                        tabular_best_prediction_df = DataHandler(
                            prediction_dictionary['best_prediction_path']).dataframe()

                        list_of_dataset += [x_test_df, y_test_df, tabular_best_prediction_df]
                elif dataset_type.endswith('Sentiment_data'):
                    prediction_dictionary_path = dataset_info['prediction_dictionary_path']
                    prediction_dictionary = DataHandler(prediction_dictionary_path).load()
                    sentiment_best_prediction_df = DataHandler(
                        prediction_dictionary['best_prediction_path']).dataframe()
                    list_of_dataset.append(sentiment_best_prediction_df)
                else:
                    raise NotImplementedError(f"Unknown dataset type: {dataset_type}")

            common_columns = find_common_columns([list_of_dataset[0], list_of_dataset[3]])
            print(f"Common columns are {common_columns}")
            new_tabular_df = list_of_dataset[0][common_columns]
            new_tabular_df['Tabular AutoML Prediction'] = list_of_dataset[2]
            new_tabular_df['target'] = list_of_dataset[1]

            merged_dataset_path = Merge([new_tabular_df, list_of_dataset[3]]).merge_all_dataset()
            self.configuration['Merged Dataset Path'] = merged_dataset_path
