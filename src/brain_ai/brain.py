from brain_ai.utils import DataLoader


class Brain:
    def __init__(self, common_column, *args, **kwargs):
        self.common_column = common_column
        self.args = args
        self.kwargs = kwargs

    def load_data(self):
        DataLoader(self.common_column, *self.args, **self.kwargs)


class ModelSelector:
    def __init__(self):
        self.dataset_list = []  # after separating the merged datasets
        self.model_list = []
        # check memory for configurations

    def select_model(self):
        pass

    def scale_dataset(self):
        pass

    def train_models(self):
        pass

    def inference(self):
        pass
