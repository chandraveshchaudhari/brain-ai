import os

from brain_automl.utilities.data_handling import DataHandler, create_directories_from_path


class Memory:

    def __init__(self, memory_directory_path=None, configuration_dict_or_path=None, project_name="BrainAutoML"):
        self.directories_created = initiate_directory_structure(memory_directory_path, name_of_the_project=project_name,
                                                                logs=True, saved_models=True, predicted_data=True,
                                                                generated_data=True)

        if type(configuration_dict_or_path) is dict:
            self.configuration = configuration_dict_or_path
            self.configuration_path = os.path.join(self.directories_created[0], "configuration.json")
            print(self.configuration_path)
            self.generate_configuration_file()

        elif type(configuration_dict_or_path) is str:
            if os.path.isfile(configuration_dict_or_path):
                self.configuration = DataHandler(configuration_dict_or_path).load()
            else:
                raise Exception(f"Configuration file can not be loaded from {configuration_dict_or_path}.")
        else:
            self.configuration = {
                'datasets': {'Tabular_data': {'path': 'path_of_tabular_data', 'target': 'target_column_name',
                                              'prediction_dictionary': 'auto',
                                              'Generated_dataset': 'auto',
                                              'test_size': 0.33},
                             'Sentiment_data': {'path': 'path_of_tabular_data',
                                                'text_column_name': "target_column_name",
                                                'prediction_dictionary': 'auto',
                                                }
                             },
                'Merged Dataset Path': "auto",
                'Underlying_models_train_test_split': 0.33}
            self.configuration_path = os.path.join(self.directories_created[0], "configuration.json")
            print(f"Configuration file not found at {self.configuration_path}. Creating a new one.")
            self.generate_configuration_file()

    def generate_configuration_file(self, output_configuration_file_path=None):
        if not output_configuration_file_path:
            DataHandler().write(self.configuration_path, data=self.configuration)
        else:
            DataHandler().write(output_configuration_file_path, data=self.configuration)

    def save_model(self, model, model_name, path_to_save_model=None):
        pass

    def save_metric(self, model_name, path_to_save_model, metric, description=None):
        """Save the metric of the model. description contain information about the data like description = [database
        name, preprocessing steps].

        Parameters
        ----------
        path_to_save_model
        model_name
        metric
        description

        Returns
        -------

        """
        if self.metrics[model_name] is list:
            self.metrics[model_name].append({'model_path': path_to_save_model, 'metric': metric,
                                             'description': description})
        else:
            self.metrics[model_name] = [{'model_path': path_to_save_model, 'metric': metric,
                                         'description': description}]

    def load_model(self, model_name=None, path_to_model=None):
        pass


def initiate_directory_structure(memory_directory_path=os.getcwd(), name_of_the_project="project_name", logs=True,
                                 saved_models=True, predicted_data=True, generated_data=True):
    memory_directory_path = os.path.join(memory_directory_path, name_of_the_project)
    create_directories_from_path(memory_directory_path)

    directories_created = [memory_directory_path]

    if logs:
        log_directory_path = os.path.join(memory_directory_path, 'Logs')
        create_directories_from_path(log_directory_path)
        directories_created.append(log_directory_path)

    if saved_models:
        saved_model_path = os.path.join(memory_directory_path, 'Saved Models')
        create_directories_from_path(saved_model_path)
        directories_created.append(saved_model_path)

    if predicted_data:
        metrics_path = os.path.join(memory_directory_path, 'Prediction Data')
        create_directories_from_path(metrics_path)
        directories_created.append(metrics_path)

    if generated_data:
        generated_data_path = os.path.join(memory_directory_path, 'Generated Data')
        create_directories_from_path(generated_data_path)
        directories_created.append(generated_data_path)

    return directories_created
