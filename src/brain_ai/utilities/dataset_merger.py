"""
Takes multiple dataset and merge them based on common column.
"""
# import logging

import pandas as pd

from brain_ai.explainability.exploratory_data_analysis import find_common_columns, is_datetime_range_element, \
    is_datetime_element, datetime_element, range_element
from brain_ai.utilities.data_handling import DataTypeInterchange


def datapoint_is_in_range(datapoint, range_datapoint):
    if is_datetime_element(datapoint):
        datapoint = datetime_element(datapoint)
    else:
        raise TypeError(f"Datapoint ({datapoint}) is not a datetime element.")
    if is_datetime_range_element(range_datapoint):
        lower_limit, upper_limit = range_element(range_datapoint)
    else:
        raise TypeError(f"Range datapoint ({range_datapoint}) is not a range element.")

    if lower_limit <= datapoint <= upper_limit:
        return True
    else:
        return False


def range_is_in_range(record1, record_2, column):
    lower_limit_1, upper_limit_1 = record1[column].split(" to ")
    lower_limit_2, upper_limit_2 = record_2[column].split(" to ")
    if upper_limit_2 >= lower_limit_1 >= lower_limit_2 and upper_limit_2 >= upper_limit_1 >= lower_limit_2:
        return True
    elif upper_limit_1 >= lower_limit_2 >= lower_limit_1 and upper_limit_1 >= upper_limit_2 >= lower_limit_1:
        return True
    else:
        return False


def checking_if_two_ranges_or_datapoints_intersect_in_two_columns(record1, record_2, column):
    # not completed and not planned to be used. Do this manually

    print(f"{record1[column]} and {record_2[column]}")
    if is_datetime_range_element(record1[column]) and is_datetime_range_element(record_2[column]):
        return range_is_in_range(record1, record_2, column)
    elif is_datetime_range_element(record1[column]) and is_datetime_element(record_2[column]):
        return datapoint_is_in_range(record_2[column], record1[column])
    elif is_datetime_element(record1[column]) and is_datetime_range_element(record_2[column]):
        return datapoint_is_in_range(record1[column], record_2[column])
    else:
        return False


def list_of_dataframe_wrt_range_column(list_of_datasets):
    list_of_range_column_dataset = []
    left_over_datasets = []

    for dataset in list_of_datasets:
        added = False
        for column in find_common_columns(list_of_datasets):
            if is_datetime_range_element(dataset[column][dataset[column].first_valid_index()]):
                list_of_range_column_dataset.append(dataset)
                added = True
        if added is False:
            left_over_datasets.append(dataset)

    return left_over_datasets + list_of_range_column_dataset


class Merge:

    def __init__(self, list_of_datasets):

        self.common_columns = find_common_columns(list_of_datasets)
        if len(self.common_columns) == 0:
            raise Exception("No common columns found in the datasets")
        self.list_of_datasets = list_of_dataframe_wrt_range_column(list_of_datasets)
        print(f"Common columns are {self.common_columns}")
        # logging.info(f"Common columns are {self.common_columns}")

        self.sorting_column_list = self.get_sorting_column_list()
        print(f"Sorting columns are {self.sorting_column_list}")

    def merge_all_dataset(self):
        """
        Merge the datasets based on common columns. The common columns are found using the common_columns method. The
        data is converted to list of records using the DataTypeInterchange class. The records are then merged based on
        the common columns. The merged records are then converted back to the original data type using the
        DataTypeInterchange class.

        Returns
        -------

        """
        # TODO: filter each dataset based on intersection within each categorical common column
        merged_data_list = DataTypeInterchange(self.list_of_datasets[0]).records
        i = 1

        for dataset in self.list_of_datasets[1:]:
            merged_data_list = self.merge(merged_data_list,
                                          DataTypeInterchange(dataset).records)

            DataTypeInterchange(merged_data_list).dataframe.to_csv(f"merged_dataset_{i}.csv", index=False)
            print(f"merged_dataset_{i}.csv created")
            i += 1

        merged_dataset_path = "merged_dataset.csv"
        DataTypeInterchange(merged_data_list).dataframe.to_csv(merged_dataset_path, index=False)
        print(f"{merged_dataset_path} created")

        return merged_dataset_path

    def get_sorting_column_list(self):
        start, mid, end = [], [], []
        for column in self.common_columns:
            if pd.api.types.is_string_dtype(self.list_of_datasets[0][column]):
                start.append(column)
            elif pd.api.types.is_datetime64_any_dtype(self.list_of_datasets[0][column]):
                mid.append(column)
            else:
                end.append(column)
        sorting_column_list = start + mid + end

        return sorting_column_list

    def condition_check_using_all_common_columns(self, record_1, record_2):
        result = False
        for column in self.sorting_column_list:
            if record_1[column] == record_2[column]:
                result = True
                # print("equal", result)
                # print(f"{column}: {record_1[column]} == {record_2[column]}")
            elif is_datetime_range_element(record_1[column]) or is_datetime_range_element(record_2[column]):
                # print("range", result)
                result = checking_if_two_ranges_or_datapoints_intersect_in_two_columns(record_1, record_2, column)
                if result is False:
                    return result
            else:
                result = False
                # print("not equal", result)
                return result
        return result

    def merge(self, record_list, record_list_2):
        # TODO: use datatype interchange to convert the records to dataframe and vice versa

        merged_record_list = []
        # i = 0
        for record_1 in record_list:

            # if i == 1000:
            #     return merged_record_list

            for record_2 in record_list_2:
                # print(f"comparing {record_1} and {record_2} ---> {self.condition_check_using_all_common_columns(
                # record_1, record_2)}")
                if self.condition_check_using_all_common_columns(record_1, record_2):
                    # print(f"{record_1},---------------------------\n {record_2}")
                    new_record = {**record_2, **record_1}
                    merged_record_list.append(new_record)
                    # print(f"merging  record, {new_record} records merged)")
                    # i += 1

        print(f"Total {len(merged_record_list)} records merged")

        return merged_record_list
