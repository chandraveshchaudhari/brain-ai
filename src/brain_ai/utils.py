
class DataLoader:
    def __init__(self, common_column, *args, **kwargs):
        self.common_column = common_column
        self.args = args
        self.kwargs = kwargs
        self.data = None

    def load_data(self):
        for data in self.args:
            self.add_datasets(data)

    def save_data_columns(self, dataframe):
        return dataframe.columns

    def add_datasets(self, data):
        pass

    def scale_dataset(self):
        pass
