import pandas as pd


class DataClean:
    def __init__(self, pandas_dataframe, target_column_name=None):
        self.pandas_dataframe = pandas_dataframe if isinstance(pandas_dataframe,
                                                               pd.DataFrame) else pd.DataFrame.from_dict(
            pandas_dataframe)
        if isinstance(pandas_dataframe, list):
            pandas_dataframe = pd.DataFrame.from_dict(pandas_dataframe)

        if target_column_name:
            pandas_dataframe = pandas_dataframe[pandas_dataframe[target_column_name].notna()]
            print(pandas_dataframe)
        if isinstance(pandas_dataframe, pd.DataFrame):
            pandas_dataframe = pandas_dataframe.to_dict('records')

        self.list_of_data_dict = dict()
        for index, row in enumerate(pandas_dataframe):
            self.list_of_data_dict[index] = row
        self.checkpoints = []
        self.row_values = dict()
        self.column_values = dict()

    def del_row(self, index=None):
        del self.list_of_data_dict[index]

    def del_column(self, column_name=None):
        for row in self.list_of_data_dict:
            del self.list_of_data_dict[row][column_name]

    def check_nan_row(self, row_index=None):
        null_sum = 0
        data_point_sum = 0
        for element in self.list_of_data_dict[row_index]:
            if pd.isnull(element):
                null_sum += 1
            else:
                data_point_sum += 1
        return null_sum, data_point_sum

    def check_nan_column(self, column_name):
        null_sum = 0
        data_point_sum = 0
        for row in self.list_of_data_dict:
            if pd.isnull(row[column_name]):
                null_sum += 1
            else:
                data_point_sum += 1
        return null_sum, data_point_sum

    def check_all_null_values(self, data=None):
        # print("check_all_null_values")
        # select default or new data
        if data:
            list_of_data_dict = data
        else:
            list_of_data_dict = self.list_of_data_dict

        # Data structure: row * column so null_values [ row_values, column_values]
        for row_index in self.list_of_data_dict:
            self.row_values[row_index] = [0, 0]
        # print("row data holder created")

        # print(next(iter(list_of_data_dict)))
        column_names = list_of_data_dict[next(iter(list_of_data_dict))].keys()
        for col in column_names:
            self.column_values[col] = [0, 0]
        # print("column values holder created")

        # loop for checking nan values

        for row in self.list_of_data_dict:
            for column_name in column_names:
                if pd.isnull(self.list_of_data_dict[row][column_name]):
                    self.row_values[row][0] += 1
                    self.column_values[column_name][0] += 1
                else:
                    self.row_values[row][1] += 1
                    self.column_values[column_name][1] += 1

        # print("all points checked", self.row_values, self.column_values)

    def get_maximum_count_of_nan_values(self, data_map=None):
        # print("get_maximum_count_of_nan_values")
        if data_map:
            data = {**data_map[0], **data_map[1]}
        else:
            data = {**self.row_values, **self.column_values}

        res = []
        maximum_nan = 0
        for key, value in data.items():

            if value[0] > maximum_nan:
                res = [{key: value}]
            elif value[0] != 0 and value[0] == maximum_nan:
                res.append({key: value})
        # print(f"all possible null values to remove--> {res}")
        return res

    def get_minimum_count_of_data_values(self, data=None):
        # print("get_minimum_count_of_data_values")
        if not data:
            data = self.get_maximum_count_of_nan_values()

        # print(data)
        minimum_data = data[0]
        for element in data:
            for key, value in element.items():
                if value[1] < list(minimum_data.values())[0][1]:
                    minimum_data = element

        return minimum_data

    def check_percentage_nan_data(self, data_map=None):
        # method to choose row, column or both
        # print("get_maximum_percentage_of_nan_values")
        if data_map:
            data = {**data_map[0], **data_map[1]}
        else:
            data = {**self.row_values, **self.column_values}

        percentage_data = dict()
        row_total = len(self.row_values)
        column_total = len(self.column_values)
        # print(f'row_total - {row_total}, column_total - {column_total}')

        for key, value in data.items():
            if type(key) is int:
                percentage_data[key] = (value[0] / column_total) * 100
            if type(key) is str:
                percentage_data[key] = (value[0] / row_total) * 100

        return percentage_data

    def get_maximum_percentage_of_nan_values(self):
        maximum = 0
        res = []
        for key, value in self.check_percentage_nan_data().items():
            if value > maximum:
                maximum = value
                res = [{key: value}]
            elif value == maximum:
                res.append({key: value})

        return res

    def remove_best_possible_row_column(self, method):
        for k, _ in self.row_or_column_to_remove(method).items():
            key = k
        # print(f"remove_best_possible_row_column- {key}")

        if type(key) is int:
            self.update_before_del(key)
            self.del_row(key)
        else:
            self.update_before_del(key)
            self.del_column(key)

    def remove_key_from_mapping(self, key, data_null_count_mapping=None):
        data = data_null_count_mapping if data_null_count_mapping else (self.row_values,
                                                                        self.column_values)
        if type(key) is int:
            del data[0][key]
        else:
            del data[1][key]

    def update_before_del(self, key):
        if type(key) is int:
            # print(f"removing -> {key}")
            self.update_column_mapping_values(key)
            self.remove_key_from_mapping(key)
        else:
            self.update_row_mapping_values(key)
            self.remove_key_from_mapping(key)

    def update_column_mapping_values(self, row_key):

        for column in self.list_of_data_dict[row_key].keys():
            if pd.isnull(self.list_of_data_dict[row_key][column]):
                (self.row_values, self.column_values)[1][column][0] -= 1
            else:
                (self.row_values, self.column_values)[1][column][1] -= 1

    def update_row_mapping_values(self, column_key):

        for row in self.list_of_data_dict:
            if pd.isnull(self.list_of_data_dict[row][column_key]):
                (self.row_values, self.column_values)[0][row][0] -= 1
            else:
                (self.row_values, self.column_values)[0][row][1] -= 1

    def row_or_column_to_remove(self, method):
        if method == 'count':
            return self.get_minimum_count_of_data_values()
        if method == 'percentage':
            return self.get_maximum_percentage_of_nan_values()[0]
        if method == 'percentage_direct':
            return {self.sorted_percentage_nan()[0][0]: self.sorted_percentage_nan()[0][1]}

    def check_nan(self, method='percentage'):
        if method == 'count':
            row_info = max((self.row_values, self.column_values)[0].values(), key=lambda x: (x[0], x[1]))
            # print(f"ROW --- null values: {row_info[0]}, data values: {row_info[1]}")
            column_info = max((self.row_values, self.column_values)[1].values(), key=lambda x: (x[0], x[1]))
            # print(f"COLUMN --- null values: {column_info[0]}, data values: {column_info[1]}")
            return max(row_info[0], column_info[0])
        if method == 'percentage':
            for value in self.get_maximum_percentage_of_nan_values()[0].values():
                return value

    def data_cleaning_status(self, method, threshold):
        if method == 'count':
            return True if self.check_nan(method) > threshold else False
        if method == 'percentage':
            return True if self.check_nan(method) > threshold else False
        if method == 'percentage_direct':
            return True if self.sorted_percentage_nan()[1] > threshold else False

    def sorted_percentage_nan(self):
        return sorted(self.check_percentage_nan_data().items(), reverse=True, key=lambda x: x[1])

    def execute(self, iteration=None, threshold=1, method='percentage'):
        # self.check_all_null_values()
        # print("execute")
        if not iteration:
            iteration = len(self.pandas_dataframe)

        for i in range(iteration):
            print(f"iteration - {i}")
            self.check_all_null_values()
            # print(len(self.list_of_data_dict))
            if self.data_cleaning_status(method, threshold):
                self.remove_best_possible_row_column(method)
            else:
                return self.list_of_data_dict
        return self.list_of_data_dict


def dict_of_dict_to_list_of_dict(dict_of_dict):
    list_of_dict = []
    for index in dict_of_dict:
        list_of_dict.append(dict_of_dict[index])
    return list_of_dict
