# https://auto.gluon.ai/stable/api/autogluon.features.html
# feature_generator = AutoMLPipelineFeatureGenerator().fit_transform(X=X_train, y=y_train)

from math import log, e
from scipy.stats import skew

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler, MaxAbsScaler


def get_datatime_type(data_string):
    if data_string.startswith("3/31/"):
        return "Q1"
    elif data_string.startswith("6/30/"):
        return "Q2"
    elif data_string.startswith("9/30/"):
        return "Q3"
    elif data_string.startswith("12/31/"):
        return "Q4"
    elif data_string.startswith("12/1/"):
        return "Y"
    else:
        print(data_string)
        return None


def logarithm_transformation_apply(number, base=e):
    if base is None:
        base = 10
    else:
        base = int(base)
    return log(number, base)


def df_apply_custom_function_on_multiple_cols(dataframe, columns_list, custom_function, **kwargs):
    for col in columns_list:
        dataframe[col] = dataframe[col].apply(custom_function, kwargs)
    return dataframe


def scaling_time_column(dataset, column_name, elements_time_format="YYYY"):
    if elements_time_format == "YYYY":
        start_year = dataset[column_name].describe()['min'] - 1
        dataset[column_name] = dataset[column_name] - start_year
        return dataset
    else:
        time_column_list = list(dataset[column_name])
        dataset[column_name] = [date_format_to_numeric_format(x) for x in time_column_list]


def engineer_datetime(pandas_datetime_column):
    # Convert the column to datetime
    pandas_datetime_column = pd.to_datetime(pandas_datetime_column)
    # Get the minimum datetime in the column
    min_datetime = pandas_datetime_column.min()

    # Decrease all of the datetimes in the column by the minimum datetime
    pandas_datetime_column = pandas_datetime_column - min_datetime

    # Convert the datetimes to floats
    pandas_datetime_column = pandas_datetime_column / np.timedelta64(1, 'D')

    return pandas_datetime_column


def single_engineer_datetime(datetime_str, min_datetime_str):
    # Convert the datetime string to a datetime object
    datetime_obj = pd.to_datetime(datetime_str)

    # Get the minimum datetime
    min_datetime = pd.to_datetime(min_datetime_str)

    # Convert the datetime to a float
    float_value = (datetime_obj - min_datetime) / np.timedelta64(1, 'D')

    return float_value


def date_format_to_numeric_format(date_format, start_date=2006):
    time_config = {"Q1": 0.2, "Q2": 0.4, "Q3": 0.6, "Q4": 0.8, 'Y': 1}
    string_date_format = str(date_format)
    last_digits = float(string_date_format.split('/')[-1])
    year_numeric_format = last_digits - start_date
    quarter_numeric_format = time_config[get_datatime_type(string_date_format)]
    return year_numeric_format + quarter_numeric_format


class FeatureEngineering:
    def __init__(self, dataset_df, target_column_name, preserve_columns=None):
        """{'StandardScaler':[], 'MinMaxScaler':[], 'logarithm_transformation_apply':[],
                                    'scaling_time_column':[], 'LabelEncoder':[]}"""
        self.dataset_df = dataset_df
        self.target_column_name = target_column_name
        self.preserve_columns = preserve_columns

        self.configuration_dictionary = self.deciding_feature_engineering_based_on_columns()


    def get_available_columns_list(self, input_column_names_list):
        available_columns = []
        for col in input_column_names_list:
            if col in self.dataset_df.columns:
                available_columns.append(col)
        return available_columns

    def deciding_feature_engineering_based_on_columns(self):
        column_scaling_type_dict = {'StandardScaler': [], 'MinMaxScaler': [],
                                    'logarithm_transformation_apply': [],  'MaxAbsScaler': [],
                                    'engineer_datetime': [], 'LabelEncoder': [], 'Delete': []}

        for column in self.dataset_df.columns:
            if column in self.preserve_columns:
                print(f"{column} is not scaled")
                continue

            if column == self.target_column_name:
                column_scaling_type_dict['LabelEncoder'].append(column)
            elif check_if_datetime_as_object_feature(self.dataset_df[column]):
                column_scaling_type_dict['engineer_datetime'].append(column)
            elif get_type_family_raw(self.dataset_df[column].dtype) in ('int', 'float'):

                if self.dataset_df[column].min() < 0:
                    column_scaling_type_dict['MaxAbsScaler'].append(column)
                elif self.dataset_df[column].min() > 10000:
                    column_scaling_type_dict['logarithm_transformation_apply'].append(column)
                elif self.dataset_df[column].min() >= 0:
                    column_scaling_type_dict['MinMaxScaler'].append(column)
            elif get_type_family_raw(self.dataset_df[column].dtype) == 'object':
                column_scaling_type_dict['Delete'].append(column)
            else:
                print(f"{column} of type ({type(self.dataset_df[column])}) is not yet implemented")
        return column_scaling_type_dict

    def scale(self):
        for function_name, column_names_list in self.configuration_dictionary.items():
            available_columns = self.get_available_columns_list(column_names_list)
            if not available_columns:
                continue

            if function_name == 'Delete':
                self.dataset_df = self.dataset_df.drop(columns=column_names_list)
            elif function_name == 'StandardScaler':
                self.dataset_df[available_columns] = StandardScaler().fit_transform(self.dataset_df[available_columns])
            elif function_name == 'MaxAbsScaler':
                self.dataset_df[available_columns] = MaxAbsScaler().fit_transform(self.dataset_df[available_columns])
            elif function_name == 'RobustScaler':
                self.dataset_df[available_columns] = RobustScaler().fit_transform(self.dataset_df[available_columns])
            elif function_name == 'MinMaxScaler':
                self.dataset_df[available_columns] = MinMaxScaler().fit_transform(self.dataset_df[available_columns])
            elif function_name == 'logarithm_transformation_apply':
                df_apply_custom_function_on_multiple_cols(self.dataset_df, available_columns,
                                                          logarithm_transformation_apply)
            elif function_name == 'engineer_datetime':
                for column in available_columns:
                    self.dataset_df[column] = engineer_datetime(self.dataset_df[column])
            elif function_name == 'LabelEncoder':
                self.dataset_df[available_columns[0]] = LabelEncoder().fit_transform(
                    list(self.dataset_df[available_columns[0]]))
            else:
                print(f"The function '{function_name}' is not yet implemented")

        return self.dataset_df


def get_type_family_raw(dtype) -> str:
    """From dtype, gets the dtype family."""
    try:
        if isinstance(dtype, pd.SparseDtype):
            dtype = dtype.subtype
        if dtype.name == "category":
            return "category"
        if "datetime" in dtype.name:
            return "datetime"
        if "string" in dtype.name:
            return "object"
        elif np.issubdtype(dtype, np.integer):
            return "int"
        elif np.issubdtype(dtype, np.floating):
            return "float"
    except Exception as err:
        print(
            f"Warning: dtype {dtype} is not recognized as a valid dtype by numpy! " f"AutoGluon may incorrectly handle this feature...")

    if dtype.name in ["bool", "bool_"]:
        return "bool"
    elif dtype.name in ["str", "string", "object"]:
        return "object"
    else:
        return dtype.name


def check_if_datetime_as_object_feature(X) -> bool:
    type_family = get_type_family_raw(X.dtype)
    # TODO: Check if low numeric numbers, could be categorical encoding!
    # TODO: If low numeric, potentially it is just numeric instead of date
    if X.isnull().all():
        return False
    if type_family != "object":  # TODO: seconds from epoch support
        return False
    try:
        # TODO: pd.Series(['20170204','20170205','20170206']) is incorrectly not detected as datetime_as_object
        #  But we don't want pd.Series(['184','822828','20170206']) to be detected as datetime_as_object
        #  Need some smart logic (check min/max values?, check last 2 values don't go >31?)
        pd.to_numeric(X)
    except:
        try:
            if len(X) > 500:
                # Sample to speed-up type inference
                X = X.sample(n=500, random_state=0)
            result = pd.to_datetime(X, errors="coerce")
            if result.isnull().mean() > 0.8:  # If over 80% of the rows are NaN
                return False
            return True
        except:
            return False
    else:
        return False


def convert_string_numeric_values_to_float(value, values_to_be_removed='%x', values_to_be_replaced=','):
    if pd.isnull(value):
        return value

    if (type(value) is float) or (type(value) is int):
        return value
    else:
        input_string = str(value)
        clean_string = input_string.rstrip(values_to_be_removed)
        stripped_string = clean_string.replace(values_to_be_replaced, '')
        try:
            float_value = float(stripped_string)
        except ValueError:
            return stripped_string
    return float_value


def header_name_change_based_on_values(dataframe, header_name, symbol_list=('x', '%'),
                                       symbol_ignore_list=('(%)', '(x)')):
    for symbol in symbol_ignore_list:
        if str(header_name).endswith(symbol):
            return dataframe

    for symbol in symbol_list:
        first_non_null = dataframe[header_name].first_valid_index()
        value = dataframe.loc[first_non_null, header_name]

        if str(value).endswith(symbol):
            percentage_header_name = f"{header_name} ({symbol})"
            dataframe.rename(columns={header_name: percentage_header_name}, inplace=True)
            return dataframe

    return dataframe


def renaming_dataset_columns_name(dataframe):
    for col in dataframe.columns:
        print(f"column:  {col}")
        dataframe = header_name_change_based_on_values(dataframe, col)
    return dataframe


def columns_union(dataset_df_1, dataset_df_2):
    all_columns = set(dataset_df_1.columns.to_list()).union(dataset_df_2.columns.to_list())
    return all_columns


def generating_column_scaling_type_dict(dataset_df, target_column_name):
    # creating configuration
    column_scaling_type_dict = {'StandardScaler': [], 'MinMaxScaler': [], 'logarithm_transformation_apply': [],
                                'scaling_time_column': [], 'LabelEncoder': [], 'Delete': []}

    all_columns = dataset_df.columns.to_list()

    for column in all_columns:
        if column == target_column_name:
            column_scaling_type_dict['LabelEncoder'].append(column)
        elif column.endswith('(x)'):
            column_scaling_type_dict['MinMaxScaler'].append(column)
        elif column.endswith('(%)'):
            column_scaling_type_dict['StandardScaler'].append(column)
        elif column == 'DateTime':
            column_scaling_type_dict['scaling_time_column'].append(column)
        elif column == 'Stock direction':
            column_scaling_type_dict['LabelEncoder'].append(column)
        elif column == 'logarithm_transformation_apply':
            column_scaling_type_dict['logarithm_transformation_apply'].append(column)
        elif type(dataset_df[column].iloc[0]) is str:
            column_scaling_type_dict['Delete'].append(column)
        else:
            print(f"{column} logarithm function is not working, so using the StandardScaler")
            column_scaling_type_dict['StandardScaler'].append(column)
    return column_scaling_type_dict
