import os

from brain_automl.explainability.exploratory_data_analysis import find_common_columns
from brain_automl.memory import Memory
from brain_automl.model_zoo.tabular_data_ai.AutoML import TabularAutoML
from brain_automl.model_zoo.text_data_ai.execution import SentimentDataExecutor
from brain_automl.utilities.data_handling import DataHandler
from brain_automl.utilities.dataset_merger import Merge, convert_date_time_to_range
from brain_automl.utilities.log_handling import Logger


class Brain(Memory):
    def __init__(self, configuration_dict_or_path=None, project_name="BrainAutoML",
                 memory_directory_path=os.getcwd(), last_months=3, *args, **kwargs):

        super().__init__(memory_directory_path=memory_directory_path,
                         configuration_dict_or_path=configuration_dict_or_path,
                         project_name=project_name)
        self.last_months = last_months
        self.project_name = project_name
        self.args = args
        self.kwargs = kwargs

        self.configuration_path = os.path.join(memory_directory_path, "configuration.json")

        self.logger = Logger(log_project_name=project_name, log_directory_path=self.directories_created[1])
        self.logger.welcome_log(project_name)
        self.logger.info(f"{project_name} started:"
                         f"Memory directory path is {self.directories_created[0]}"
                         f"logs directory path is {self.directories_created[1]}"
                         f"saved models directory path is {self.directories_created[2]}"
                         f"predicted data directory path is {self.directories_created[3]}"
                         f"generated data directory path is {self.directories_created[4]}")

    def train_and_test_brain(self, numerical_analysis_data=False):
        self.training_and_testing_subpart_of_brain()
        if 'Merged Dataset Path' not in self.configuration:
            self.merge_dataset(self.last_months)
        print(f"Reached here and value of split_data_by_column_name_and_value_dict is {self.configuration['Underlying_models_train_test_split']}")
        if type(self.configuration['Underlying_models_train_test_split']) is dict:
            brain = TabularAutoML(self.configuration['Merged Dataset Path'],
                                  self.configuration['target'],
                                  split_data_by_column_name_and_value_dict=self.configuration['Underlying_models_train_test_split'],
                                  logger=self.logger,
                                  tabular_directory=self.directories_created[0],
                                  project_name='FinalAnalysisModel',
                                  categorical_columns=self.configuration['categorical_columns']
                                  )
        else:
            brain = TabularAutoML(self.configuration['Merged Dataset Path'],
                                  self.configuration['target'],
                                  logger=self.logger,
                                  tabular_directory=self.directories_created[0],
                                  project_name='FinalAnalysisModel',
                                  categorical_columns=self.configuration['categorical_columns']
                                  )
        brain.train_predict_save_metrics()
        prediction_path = os.path.join(brain.saved_models_directory_path, 'performance_metrics.csv')
        self.configuration['prediction_path'] = prediction_path
        self.generate_configuration_file()
        return brain.performance_metrics()

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
                                                                  tabular_directory=self.directories_created[0],
                                                                  categorical_columns=dataset_info['categorical_columns'])
                            dataset_info['Directories Created'] = tabular_automl_object.directories_created
                            tabular_automl_object.train_predict_save_metrics()
                            tabular_automl_object.best_model_prediction_path()
                            self.configuration['datasets'][dataset_type]['prediction_dictionary_path'] = os.path.join(
                                dataset_info['Directories Created'][3], 'prediction_dictionary.json')
                            self.generate_configuration_file()

                        elif 'test_size' in dataset_info:
                            tabular_automl_object = TabularAutoML(dataset_info['path'], dataset_info['target'],
                                                                  test_size=dataset_info['test_size'],
                                                                  logger=self.logger,
                                                                  tabular_directory=self.directories_created[0],
                                                                  categorical_columns=dataset_info['categorical_columns'])
                            dataset_info['Directories Created'] = tabular_automl_object.directories_created
                            tabular_automl_object.train_predict_save_metrics()
                            tabular_automl_object.best_model_prediction_path()
                            self.configuration['datasets'][dataset_type]['prediction_dictionary_path'] = os.path.join(
                                dataset_info['Directories Created'][3], 'prediction_dictionary.json')
                            self.generate_configuration_file()

                        else:
                            tabular_automl_object = TabularAutoML(dataset_info['path'], dataset_info['target'],
                                                                  logger=self.logger,
                                                                  tabular_directory=self.directories_created[0],
                                                                  categorical_columns=dataset_info['categorical_columns'])
                            dataset_info['Directories Created'] = tabular_automl_object.directories_created
                            tabular_automl_object.train_predict_save_metrics()
                            tabular_automl_object.best_model_prediction_path()
                            self.configuration['datasets'][dataset_type]['prediction_dictionary_path'] = os.path.join(
                                dataset_info['Directories Created'][3], 'prediction_dictionary.json')
                            self.generate_configuration_file()
                elif dataset_type.endswith('Sentiment_data'):
                    if 'prediction_dictionary_path' not in dataset_info:
                        sentiment_data = SentimentDataExecutor(dataset_info['path'],
                                                               dataset_info['target']).add_result_column()
                        sentiment_data_path = os.path.join(self.directories_created[4], 'sentiment_data.csv')
                        dataset_info['y_pred'] = sentiment_data_path
                        DataHandler().write(sentiment_data_path, sentiment_data)
                        self.configuration['datasets'][dataset_type]['prediction_dictionary_path'] = os.path.join(
                            dataset_info['Directories Created'][3], 'prediction_dictionary.json')
                        self.generate_configuration_file()
                else:
                    raise NotImplementedError(f"Unknown dataset type: {dataset_type}")

    def merge_dataset(self, last_months=3):
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
                        print(prediction_dictionary_path)
                        tabular_best_prediction_df = DataHandler(
                            prediction_dictionary['best_prediction_path']).dataframe()
                        # take the first column of tabular_best_prediction_df
                        tabular_best_prediction_df = tabular_best_prediction_df.iloc[:, 0]

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
            print(new_tabular_df)
            # convert DateTime to range Quarter
            # todo: add the functionality to find the date time columns
            new_tabular_df[common_columns[0]] = new_tabular_df[common_columns[0]].map(
                lambda x: convert_date_time_to_range(x, last_months))


            new_tabular_df['Tabular AutoML Prediction'] = list_of_dataset[2]
            # new_tabular_df['target'] = list_of_dataset[1]
            print(new_tabular_df)
            print(list_of_dataset[3])

            merged_dataset_path = Merge([new_tabular_df, list_of_dataset[3]]).merge_all_dataset()
            self.configuration['Merged Dataset Path'] = merged_dataset_path
            self.generate_configuration_file()
            print("added merged file path in configuration.")
