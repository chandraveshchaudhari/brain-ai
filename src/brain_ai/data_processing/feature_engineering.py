from math import log

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

from data_processing.data_creation import get_datatime_type


def logarithm_transformation_apply(number, base=None):
    # NaNs are treated as missing values: disregarded in fit, and maintained in transform.
    if not number:
        return None

    if number in (-1, 0, 1):
        return 0
    negative = False if number >= 0 else True

    absolute_value = abs(number) if negative else number
    logarithm_value = log(absolute_value, base) if base else log(absolute_value)

    return -1 * logarithm_value if negative else logarithm_value


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


def date_format_to_numeric_format(date_format, start_date=2006):
    time_config = {"Q1": 0.2, "Q2": 0.4, "Q3": 0.6, "Q4": 0.8, 'Y': 1}
    string_date_format = str(date_format)
    last_digits = float(string_date_format.split('/')[-1])
    year_numeric_format = last_digits - start_date
    quarter_numeric_format = time_config[get_datatime_type(string_date_format)]
    return year_numeric_format + quarter_numeric_format


class FeatureEngineering:
    def __init__(self, dataset_df, configuration_dictionary):
        """{'StandardScaler':[], 'MinMaxScaler':[], 'logarithm_transformation_apply':[],
                                    'scaling_time_column':[], 'LabelEncoder':[]}"""
        self.configuration_dictionary = configuration_dictionary
        self.dataset_df = dataset_df

    def get_available_columns_list(self, input_column_names_list):
        available_columns = []
        for col in input_column_names_list:
            if col in self.dataset_df.columns:
                available_columns.append(col)
        return available_columns

    def scale(self):
        for function_name, column_names_list in self.configuration_dictionary.items():
            available_columns = self.get_available_columns_list(column_names_list)
            if not available_columns:
                continue

            if function_name == 'Delete':
                self.dataset_df = self.dataset_df.drop(columns=column_names_list)
            elif function_name == 'StandardScaler':
                self.dataset_df[available_columns] = StandardScaler().fit_transform(self.dataset_df[available_columns])
            elif function_name == 'MinMaxScaler':
                self.dataset_df[available_columns] = MinMaxScaler().fit_transform(self.dataset_df[available_columns])
            elif function_name == 'logarithm_transformation_apply':
                df_apply_custom_function_on_multiple_cols(self.dataset_df, available_columns,
                                                          logarithm_transformation_apply)
            elif function_name == 'scaling_time_column':
                scaling_time_column(self.dataset_df, available_columns[0], 'DD/MM/YYYY')
            elif function_name == 'LabelEncoder':
                self.dataset_df[available_columns[0]] = LabelEncoder().fit_transform(list(self.dataset_df[available_columns[0]]))

            else:
                print(f"The function '{function_name}' is not yet implemented")

        return self.dataset_df


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


def generating_column_scaling_type_dict(dataset_df):
    # creating configuration
    column_scaling_type_dict = {'StandardScaler': [], 'MinMaxScaler': [], 'logarithm_transformation_apply': [],
                                'scaling_time_column': [], 'LabelEncoder': [], 'Delete': []}

    all_columns = dataset_df.columns.to_list()

    for column in all_columns:
        if column.endswith('(x)'):
            column_scaling_type_dict['MinMaxScaler'].append(column)
        elif column.endswith('(%)'):
            column_scaling_type_dict['StandardScaler'].append(column)
        elif column == 'DateTime':
            column_scaling_type_dict['scaling_time_column'].append(column)
        elif column == 'Stock direction':
            column_scaling_type_dict['LabelEncoder'].append(column)
        elif type(dataset_df[column].iloc[0]) is str:
            column_scaling_type_dict['Delete'].append(column)
        else:
            print(f"{column} logarithm function is not working, so using the StandardScaler")
            column_scaling_type_dict['StandardScaler'].append(column)
    return column_scaling_type_dict


