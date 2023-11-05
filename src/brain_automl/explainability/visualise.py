import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def graph_all_columns(dataframe, number_of_columns=1, figsize=(20, 60)):
    """not working properly"""
    number_of_rows = len(dataframe.columns) // number_of_columns

    fig, axes = plt.subplots(nrows=number_of_rows, ncols=number_of_columns, figsize=figsize)

    dataframe.plot(subplots=True, ax=axes)

    plt.show()


def correlation_diagram(dataset, figure_size=(30, 30)):
    fig, ax = plt.subplots(figsize=figure_size)
    # plotting the heatmap for correlation
    ax = sns.heatmap(dataset.corr(), center=0)
    plt.show()


def create_overleaf_longtable_column_premble(number_of_columns):
    start = '|'
    mid = "l|" * number_of_columns
    return start + mid


def create_overleaf_longtable_column_description(dataframe):
    final_string = ""
    list_of_column = dataframe.columns.to_list()
    for i in range(len(list_of_column)):
        if i == 0:
            if len(list_of_column) == 1:
                final_string += r"\multicolumn{1}{|c|}{\textbf{" + f"{list_of_column[i]}" + r"}} "
            else:
                final_string += r"\multicolumn{1}{|c|}{\textbf{" + f"{list_of_column[i]}" + r"}} & "
        elif i == len(list_of_column) - 1:
            final_string += r"\multicolumn{1}{c|}{\textbf{" + f"{list_of_column[i]}" + r"}} "
        else:
            final_string += r"\multicolumn{1}{c|}{\textbf{" + f"{list_of_column[i]}" + r"}} & "

    return final_string


def escape_latex_sensitive_characters(string):
    if type(string) is not str:
        return string

    string = string.replace("\\", "\\\\")  # this is a special case \ must be the first one
    string = string.replace(r"_", r"\_")
    string = string.replace(r"#", r"\#")
    string = string.replace(r"%", r"\%")
    string = string.replace(r"&", r"\&")
    string = string.replace(r"{", r"\{")
    string = string.replace(r"}", r"\}")
    string = string.replace(r"~", r"\~")
    string = string.replace(r"^", r"\^")
    string = string.replace(r"$", r"\$")
    return string


def scale_numbers(number):
    if type(number) is str:
        return number
    return round(number, 6)


def converting_main_content(dataframe, number_scale=6):
    final_string = ""
    for i in range(len(dataframe)):
        for j in range(len(dataframe.columns)):
            if j == 0:
                if j == len(dataframe.columns) - 1:
                    new_string = f"{scale_numbers(escape_latex_sensitive_characters(dataframe.iloc[i, j]))}" + " \\\\ \n"
                    final_string += new_string
                else:
                    final_string += f"{scale_numbers(escape_latex_sensitive_characters(dataframe.iloc[i, j]))} & "

            elif j == len(dataframe.columns) - 1:
                final_string += f"{scale_numbers(escape_latex_sensitive_characters(dataframe.iloc[i, j]))} \\\\ \n"
            else:
                final_string += f"{scale_numbers(escape_latex_sensitive_characters(dataframe.iloc[i, j]))} & "
    return final_string


def convert_pandas_dataframe_to_overleaf_longtable_string(pandas_df, caption, label,
                                                          information="Source: compiled by researcher", detailed=True,
                                                          number_scale=6):
    if detailed:
        pd.set_option('display.max_colwidth', None)
    else:
        pd.set_option('display.max_colwidth', 50)

    number_of_columns = len(pandas_df.columns)
    longtable_begin = r"\begin{center}"
    first = r"\begin{longtable}{" + f"{create_overleaf_longtable_column_premble(number_of_columns)}" + "}"
    second = r"\caption{" + f"{caption}" + "}"
    third = r"\label{" + f"tab:{label}" + "}" + r"\\"
    fourth = r"\hline " + " \n " + f"{create_overleaf_longtable_column_description(pandas_df)}" + r" \\" + " \n " + r"\hline "
    fifth = r"\endfirsthead" + " \n \n "
    sixth = r"\multicolumn{" +f"{number_of_columns}" + r"}{c}" + "\n" + r"{{\bfseries \tablename\ \thetable{} -- continued from previous page}} \\"
    seventh = r"\hline " + " \n " + f"{create_overleaf_longtable_column_description(pandas_df)}" + r"\\ " + " \n " +r"\hline "
    eighth = r"\endhead" + " \n \n "
    ninth = r"\hline" + " \n" + r"\multicolumn{" + f"{number_of_columns}" + r"}{|r|}" + r"{{Continued on next page}} \\ " + " \n " +r"\hline "
    tenth = r"\endfoot" + " \n \n "
    eleventh = r"\hline \hline"
    twelfth = r"\endlastfoot" + " \n \n "

    main_content = f"{converting_main_content(pandas_df, number_scale=number_scale)}"
    end_longtable = r"\end{longtable}" + "\n" + f"{information}" + "\n" + r"\end{center}"

    string_to_return = (longtable_begin + "\n" + first + "\n" + second + "\n" + third + "\n" + fourth + "\n" + fifth +
                        "\n" + sixth + "\n" + seventh + "\n" + eighth + "\n" + ninth + "\n" + tenth + "\n" + eleventh +
                        "\n" + twelfth + "\n" + main_content + "\n" + end_longtable)
    return string_to_return


def convert_pandas_dataframe_to_overleaf_longtable_string_to_tex_file(pandas_df, filename, caption, label,
                                                                      information="Source: compiled by researcher",
                                                                      detailed=True, number_scale=6):
    string = convert_pandas_dataframe_to_overleaf_longtable_string(pandas_df, caption=caption, label=label,
                                                                   information=information, detailed=detailed,
                                                                   number_scale=number_scale)
    with open(filename, 'w') as f:
        f.write(string)
