"""
Takes multiple dataset and merge them based on common column.
"""
import pandas as pd

from brain_ai.utilities.data_handling import first_valid_pandas_column_data


def find_common_elements(*lists):
    if len(lists) < 2:
        return []

    common_elements = set(lists[0])

    for lst in lists[1:]:
        common_elements = common_elements.intersection(set(lst))

    return list(common_elements)



class Merge:
    def __init__(self, list_of_datasets):
        self.list_of_datasets = list_of_datasets
        self.common_columns = find_common_elements(*[dataset.columns for dataset in self.list_of_datasets])
        self.sorting_column_list = self.get_sorting_column_list()

    def dataset_sorted_by_sorting_columns_list(self):
        return [dataset.sort_values(by=self.sorting_column_list) for dataset in self.list_of_datasets]

    def get_sorting_column_list(self):
        start, mid, end = [], [], []
        for column in self.common_columns:
            if pd.api.types.is_string_dtype(self.list_of_datasets[0][column]):
                start.append(column)
            elif pd.api.types.is_datetime64_any_dtype(self.list_of_datasets[0][column]):
                mid.append(column)
            else:
                end.append(column)
        return start + mid + end

    def longest_dataset(self):
        return max(self.list_of_datasets, key=len)

    def descending_list_of_dataframe_by_length(self):
        return sorted(self.list_of_datasets, key=len, reverse=True)

    def merge(self):
        """
        Merge the datasets based on common columns. The common columns are found using the common_columns method. The
        data is converted to list of records using the DataTypeInterchange class. The records are then merged based on
        the common columns. The merged records are then converted back to the original data type using the
        DataTypeInterchange class.

        Returns
        -------

        """
        merged_data_records = []

        index = 0

        while index < len(self.descending_list_of_dataframe_by_length()[0]):
            for dataset in self.descending_list_of_dataframe_by_length():
                for column in self.sorting_column_list:














