"""This module is responsible for handling the data."""
import json
import logging
import os
from typing import Any

import pandas as pd


# Data Handler functions
def write_json_file(output_file_path: str, data: Any) -> None:
    """Write json file at output_file_path with the help of input dictionary.
    Parameters
    ----------

    output_file_path : str
        This is the path of output file we want, if only name is provided then it will export json to the script path.
    data : Any
        This is the python dictionary which we want to be saved in json file format.
    Returns
    -------
    None
        Function doesn't return anything but write a json file at output_file_path.
    """
    with open(output_file_path, "w") as outfile:
        json.dump(data, outfile, indent=4)


def read_json_file(input_file_path: str) -> Any:
    """Read json file at input_file_path and return the data.
    Parameters
    ----------
    input_file_path : str
        This is the path of input file we want to read.
    Returns
    -------
    Any
        Function returns the data of json file at input_file_path.
    """
    with open(input_file_path, "r") as infile:
        data = json.load(infile)
    return data


def append_to_json_file(output_file_path: str, data: Any) -> None:
    """Write json file at output_file_path with the help of input dictionary.
    Parameters
    ----------

    output_file_path : str
        This is the path of output file we want, if only name is provided then it will export json to the script path.
    data : Any
        This is the python dictionary which we want to be saved in json file format.
    Returns
    -------
    None
        Function doesn't return anything but write a json file at output_file_path.
    """
    with open(output_file_path, "a") as outfile:
        json.dump(data, outfile)


class DataHandler:
    """
        This class is responsible for loading the data from the given path without specifying the type of data.
    """

    def __init__(self, data_path=None, *args, **kwargs):
        self.data_path = data_path
        self.args = args
        self.kwargs = kwargs
        self.data = self.load() if data_path else None

    def load(self):
        if self.data_path.endswith('.csv'):
            return pd.read_csv(self.data_path, low_memory=False)
        elif self.data_path.endswith('.xlsx'):
            return pd.read_excel(self.data_path)
        elif self.data_path.endswith('.json'):
            return read_json_file(self.data_path)

    def write(self, output_file_path=None, data=None):
        output_file_path = self.data_path if output_file_path is None else output_file_path
        data_to_write = data if data is not None else self.data
        if data_to_write is pd.DataFrame:
            if output_file_path.endswith('.csv'):
                return data_to_write.to_csv(output_file_path)
            elif output_file_path.endswith('.xlsx'):
                return data_to_write.to_excel(output_file_path)
        elif output_file_path.endswith('.json'):
            return write_json_file(output_file_path, data_to_write)

    def dataframe(self):
        if type(self.data) is pd.DataFrame:
            return self.data

        if type(self.data) is dict:
            self.data = DataTypeInterchange(self.data).dataframe
        return self.data

    def records(self):
        if type(self.data) is pd.DataFrame:
            self.data = DataTypeInterchange(self.data).records
        return self.data


class DataTypeInterchange:
    """Using this class we can interchange the data type from one to another.(dict "records", dataframe)"""

    def __init__(self, data):
        self.data = data
        logging.info(f"Data type is {type(self.data)}")

    @property
    def dataframe(self):
        # Create a DataFrame from the dictionary
        self.data = pd.DataFrame.from_dict(self.data)
        return self.data  # return the dataframe

    @property
    def records(self):
        self.data = self.data.to_dict('records')
        return self.data  # return the records

    @property
    def cache_dict(self):
        self.data = self.data.to_dict()
        return self.data


def first_valid_pandas_column_data(dataframe, column):
    valid_index = dataframe[column].first_valid_index()
    return dataframe[column][valid_index]
# end


def specify_dataset_type():
    pass


def create_file_if_not_present(file_path, file_creation_function):
    """
    This function is used to create a file if it is not present.
    Parameters
    ----------
    data
    file_creation_function
    file_path : str
        The path of the file to be created.

    Returns
    -------

    """
    if os.path.isfile(file_path):
        print("File already present")
    else:
        print("Creating file")
        file_creation_function(file_path, [])


def create_directories_from_path(path):
    dir_path = os.path.dirname(path)
    # Check if the path exists
    if not os.path.exists(dir_path):
        # Create all directories in the path
        os.makedirs(dir_path)


class DatasetCreator:
    # creates the dataset by running models on the data
    pass