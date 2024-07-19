from datetime import datetime


def find_common_columns(dataframes):
    if len(dataframes) < 2:
        return []

    common_columns = set(dataframes[0].columns)

    for dataframe in dataframes[1:]:
        common_columns = common_columns.intersection(set(dataframe.columns))

    if 'DateTime' in common_columns:
        common_columns.remove('DateTime')
    result = ['DateTime']
    result.extend(list(common_columns))
    return result


def is_datetime_range_element(text):
    if isinstance(text, str) and " to " in text:
        lower, upper = text.split(" to ")
        if is_datetime_element(lower) and is_datetime_element(upper):
            return True

    return False


def range_element(text):
    lower, upper = text.split(" to ")
    return datetime_element(lower), datetime_element(upper)


def is_datetime_element(text, split_character="-"):
    try:
        datetime_element(text, split_character)
        return True
    except:
        return False


def datetime_element(text, split_character="-"):
    year, month, day = text.split(split_character)
    return datetime(year=int(year), month=int(month), day=int(day))
