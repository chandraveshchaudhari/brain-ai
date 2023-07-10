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
