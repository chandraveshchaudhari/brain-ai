# import logging

from brain_ai.memory import Memory
from brain_ai.model_zoo.model_selector_and_trainer import ModelZoo
from brain_ai.model_zoo.tabular_data_ai.execution import TabularAIExecutor
from brain_ai.utilities.data_handling import DataHandler
from brain_ai.utilities.dataset_merger import Merge


class Brain(Memory):
    def __init__(self, memory_directory_path=None, *args, **kwargs):
        super().__init__(memory_directory_path)
        self.merged_dataset_path = None
        self.args = args
        self.kwargs = kwargs

    def merge_dataset(self):
        # TODO: add the functionality to merge the dataset (too complicated to do it for unknown data types)

        if 'merged_dataset' in self.configuration and self.configuration['merged_dataset']:
            self.merged_dataset_path = self.configuration['merged_dataset']
        else:
            list_of_dataset = []
            for data_type_and_data in self.configuration['datasets']:
                for _, data in data_type_and_data.items():
                    list_of_dataset.append(DataHandler(data['path']).dataframe())
                    # print(f"merging the dataset {data['path']} into list_of_dataset")
                    # logging.info(f"merging the dataset {data['path']} into list_of_dataset")
            self.merged_dataset_path = Merge(list_of_dataset).merge_all_dataset()

    def base_model_train(self):
        # logging.info("Training the models")
        metric = ModelZoo(self.configuration).base_models_train_and_test()
        return metric

    def train(self):
        # logging.info("Training the models")
        metric = TabularAIExecutor(DataHandler(self.merged_dataset_path).dataframe(),
                                   self.configuration['target'])
        return metric

    def inference(self):
        # get the saved models from the memory and use them to predict the new data.
        pass

#
# if __name__ == "__main__":
#     my_brain = Brain("/home/chandravesh/PhDWork/JupyterProjects/brain-ai-jupyter-notebooks/")
#     my_brain.merge_dataset()
