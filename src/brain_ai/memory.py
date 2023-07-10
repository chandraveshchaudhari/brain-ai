import os

from brain_ai.utilities.data_handling import DataHandler


class Memory:

    def __init__(self, memory_directory_path=None):
        self.configuration_path = "./configuration.json" if memory_directory_path is None else os.path.join(
            memory_directory_path, "configuration.json")

        self.configuration = DataHandler(self.configuration_path).load() if memory_directory_path else {"datasets": [
            {'Tabular_data': {'path': 'path_of_tabular_data', 'target': 'target_column_name', 'train_test_split': 0.2}
             },
            {'Tabular_data': {'path': 'path_of_tabular_data', 'target': 'target_column_name'}
             },
            {'Sentiment_data': {'path': 'path_of_tabular_data', 'target': 'target_column_name'}
             }],
            "Underlying_models_train_test_split": 0.2, }

    def generate_configuration_file(self, output_configuration_file_path=None):
        DataHandler(output_configuration_file_path).write(data=self.configuration)
