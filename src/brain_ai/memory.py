import os
import sys

sys.path.append("/home/chandravesh/PhDWork/PycharmProjects/data-processing/src")
from data_processing import file_paths, utils as data_processing_utils


class Memory:

    def __init__(self, memory_path=file_paths.FilePaths().Sentiment_Analysis_Metric_directory_path):
        self.memory_path = memory_path
        self.configuration_json_path = os.path.join(self.memory_path, "configuration.json")
        self.configuration = {"dataset_type": "model_name"}

    def add_in_configuration(self, data):
        data_processing_utils.append_to_json_file(data, self.configuration_json_path)
