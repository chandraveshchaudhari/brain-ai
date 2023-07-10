"""
Takes multiple dataset and merge them based on common column.
"""
import pandas as pd

from brain_ai.explainability.exploratory_data_analysis import find_common_columns
from brain_ai.utilities.data_handling import DataTypeInterchange


def is_in_range(datapoint, range_datapoint):
    assert isinstance(range_datapoint, str), "range_datapoint should be a string"

    lower_limit, upper_limit = range_datapoint.split(" to ")
    if lower_limit <= datapoint <= upper_limit:
        return True
    else:
        return False


class Merge:

    def __init__(self, list_of_datasets):
        self.list_of_datasets = list_of_datasets
        self.common_columns = find_common_columns(self.list_of_datasets)
        self.descending_list_of_dataframe_by_length = list(
            sorted(self.list_of_datasets, key=lambda x: len(x), reverse=True))

    def merge_all_dataset(self):
        """
        Merge the datasets based on common columns. The common columns are found using the common_columns method. The
        data is converted to list of records using the DataTypeInterchange class. The records are then merged based on
        the common columns. The merged records are then converted back to the original data type using the
        DataTypeInterchange class.

        Returns
        -------

        """
        merged_records_list = DataTypeInterchange(self.descending_list_of_dataframe_by_length[0]).records

        for dataset in self.descending_list_of_dataframe_by_length[1:]:
            record_list = DataTypeInterchange(dataset).records
            merged_records_list = self.merge(merged_records_list, record_list)

        return merged_records_list

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

    def condition_check_using_all_common_columns(self, record1, record_2):
        for column in self.get_sorting_column_list():
            if isinstance(record1[column], str) and record1[column].lower().contains(" to "):
                if is_in_range(record_2[column], record1[column]):
                    pass
                else:
                    return False
            elif isinstance(record_2[column], str) and record_2[column].lower().contains(" to "):
                if is_in_range(record1[column], record_2[column]):
                    pass
                else:
                    return False
            elif record1[column] == record_2[column]:
                pass

        return False

    def merge(self, record_list, record_list_2):
        merged_record_list = []
        for record in record_list:
            for record_2 in record_list_2:
                if self.condition_check_using_all_common_columns(record, record_2):
                    merged_record_list.append({**record, **record_2})

        return merged_record_list
