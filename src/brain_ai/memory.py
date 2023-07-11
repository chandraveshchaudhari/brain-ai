import os

from brain_ai.utilities.data_handling import DataHandler


class Memory:

    def __init__(self, memory_directory_path=None):

        if os.path.exists(memory_directory_path):
            configuration_path = os.path.join(memory_directory_path, "configuration.json")
            if os.path.isfile(configuration_path):
                self.configuration = DataHandler(configuration_path).load()

        else:
            self.configuration = {"datasets": [
                {'Tabular_data': {'path': 'path_of_tabular_data', 'target': 'target_column_name', 'test_size': 0.2}
                 },
                {'Sentiment_data': {'path': 'path_of_tabular_data'}
                 }],
                "Underlying_models_train_test_split": 0.2, }

    def generate_configuration_file(self, output_configuration_file_path=None):
        DataHandler().write(output_configuration_file_path, data=self.configuration)
