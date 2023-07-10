

def find_common_columns(*dataframes):
    if len(dataframes) < 2:
        return []

    common_columns = set(dataframes[0].columns)

    for dataframe in dataframes[1:]:
        common_columns = common_columns.intersection(set(dataframe.columns))

    return list(common_columns)
