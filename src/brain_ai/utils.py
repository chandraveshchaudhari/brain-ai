def specify_dataset_type():
    pass


class DataLoader:
    def __init__(self, common_column, *args, **kwargs):
        self.common_column = common_column
        self.args = args
        self.kwargs = kwargs
        self.data = []

    def load_data(self):
        for dataset in self.args:
            self.data.append(dataset)

    def data_handler(self):
        specify_dataset_type()  # save configurations in memory

    def merge_all_datasets_using_common_column(self):
        pass

    # end of file

    def save_data_columns(self, dataframe):
        return dataframe.columns

    def add_datasets(self, data):
        pass

    def scale_dataset(self):
        pass
