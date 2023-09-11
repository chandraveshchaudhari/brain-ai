"""This module is responsible for handling the data."""
import json
import logging
import os
import pickle
from typing import Any

import numpy as np
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


class SaveData:
    def __init__(self, save_directory_path=os.getcwd()):
        self.save_directory_path = save_directory_path

    def save(self, data, data_name):
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
            file_name = f"{data_name}.csv"
            file_path = os.path.join(self.save_directory_path, file_name)
            data.to_csv(file_path)
        elif type(data) is pd.DataFrame:
            file_name = f"{data_name}.csv"
            file_path = os.path.join(self.save_directory_path, file_name)
            data.to_csv(file_path)
        else:
            save_to_pickle(os.path.join(self.save_directory_path, f"{data_name}.pkl"), data)


class DataTypeInterchange:
    """Using this class we can interchange the data type from one to another.(dict "records", dataframe)"""

    def __init__(self, data):
        if type(data) is pd.DataFrame:
            self.dataframe = data
            self.records = self.dataframe.to_dict('records')
        elif type(data) is list:
            self.dataframe = pd.DataFrame.from_dict(data)
            self.records = data
        else:
            logging.error(f"Data type ({type(data)}) is not supported.")

    def dataframe_to_records(self):
        self.records = self.dataframe.to_dict('records')
        return self.records

    def records_to_dataframe(self):
        self.dataframe = pd.DataFrame.from_dict(self.records)
        return self.dataframe


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


class DataPaths:
    def __init__(self, config_location=None):
        self.config_location = config_location
        self.paths = read_json_file(self.config_location) if os.path.isfile(self.config_location) else {}

    def save_file_paths(self):
        file_format = self.config_location.split(".")[-1]

        if file_format == "json":
            write_json_file(self.config_location, self.paths)
        else:
            raise NotImplementedError(f"File format {file_format} is not implemented yet")

    def update(self, data_name_and_path_mapping):
        create_directories_from_path(list(data_name_and_path_mapping.values())[0])
        self.paths.update(data_name_and_path_mapping)
        # Sort the dictionary by values
        sorted_data_by_values = {k: v for k, v in sorted(self.paths.items(), key=lambda item: item[1])}
        self.paths = sorted_data_by_values

        self.save_file_paths()


def create_directory(directory_name):
    """Creates a directory if it does not exist."""
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)


def save_to_pickle(file_name, data):
    try:
        with open(file_name, 'wb') as file:
            pickle.dump(data, file)
        print(f'Data saved to {file_name}')
    except Exception as e:
        print(f'Error saving data to {file_name}: {str(e)}')


# Function to load a Python object from a pickle file
def load_from_pickle(file_name):
    try:
        with open(file_name, 'rb') as file:
            loaded_data = pickle.load(file)
        print(f'Data loaded from {file_name}')
        return loaded_data
    except Exception as e:
        print(f'Error loading data from {file_name}: {str(e)}')
        return None
