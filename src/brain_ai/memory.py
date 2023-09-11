import os

from brain_ai.utilities.data_handling import DataHandler


class Memory:

    def __init__(self, memory_directory_path=None):
        if not memory_directory_path:
            self.memory_directory_path = os.path.join(os.getcwd(), "memory")
        else:
            self.memory_directory_path = memory_directory_path

        if not os.path.exists(self.memory_directory_path):
            os.makedirs(self.memory_directory_path)
        self.configuration_path = os.path.join(self.memory_directory_path, "configuration.json")
        if os.path.isfile(self.configuration_path):
            self.configuration = DataHandler(self.configuration_path).load()
        else:
            self.configuration = {'metrics path': os.path.join(self.memory_directory_path, "metrics"),
                                  "datasets":
                                      {'Tabular_data': {'path': 'path_of_tabular_data', 'target': 'target_column_name',
                                                        'test_size': 0.2}
                                          ,
                                       'Sentiment_data': {'path': 'path_of_tabular_data'}}
                ,
                                  "Underlying_models_train_test_split": 0.2
                                  }
        if not os.path.exists(self.configuration['metrics path']):
            os.makedirs(self.configuration['metrics path'])
            self.metrics = dict()
        else:
            self.metrics = DataHandler(self.configuration['metrics path']).load()

    def generate_configuration_file(self, output_configuration_file_path=None):
        if not output_configuration_file_path:
            DataHandler().write(self.configuration_path, data=self.configuration)
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
