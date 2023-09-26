import json
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from supervised.utils.additional_metrics import AdditionalMetrics


from sklearn.metrics import precision_recall_curve, roc_curve, class_likelihood_ratios, det_curve
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.metrics import hinge_loss, matthews_corrcoef, roc_auc_score, top_k_accuracy_score
from sklearn.metrics import accuracy_score, classification_report, f1_score, fbeta_score
from sklearn.metrics import hamming_loss, jaccard_score, log_loss, multilabel_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score
from sklearn.metrics import zero_one_loss, average_precision_score


def calculate_all_classification_metrics(y_true, y_pred, y_scores=None, pos_label=None):

    metrics_dict = {}

    # Precision-Recall Curve
    if y_scores is not None:
        precision, recall, _ = precision_recall_curve(y_true, y_scores, pos_label=pos_label)
        metrics_dict['Precision-Recall Curve'] = {'Precision': precision, 'Recall': recall}

        fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=pos_label)
        metrics_dict['ROC Curve'] = {'FPR': fpr, 'TPR': tpr}

        # DET Curve (Binary Classification)
        if len(set(y_true)) == 2:
            fpr_det, fnr_det, _ = det_curve(y_true, y_scores, pos_label=pos_label)
            metrics_dict['DET Curve'] = {'FPR_DET': fpr_det, 'FNR_DET': fnr_det}

        # Hinge Loss (Binary Classification)
            hinge_loss_val = hinge_loss(y_true, y_scores)
            metrics_dict['Hinge Loss'] = hinge_loss_val


            top_k_accuracy = top_k_accuracy_score(y_true, y_scores)
            metrics_dict['Top-k Accuracy'] = top_k_accuracy

                # Log Loss (Binary Classification)
            log_loss_val = log_loss(y_true, y_scores)
            metrics_dict['Log Loss'] = log_loss_val

            # ROC AUC Score (Multiclass Classification)
            roc_auc = roc_auc_score(y_true, y_scores, average='macro')
            metrics_dict['ROC AUC Score'] = roc_auc
            # Average Precision Score (Multiclass Classification)
            avg_precision = average_precision_score(y_true, y_scores, average='macro')
            metrics_dict['Average Precision Score'] = avg_precision

        # ROC AUC Score

        roc_auc = roc_auc_score(y_true, y_scores)
        metrics_dict['ROC AUC Score'] = roc_auc



    # Class Likelihood Ratios (Binary Classification)
    if len(set(y_true)) == 2:
        lr_pos, lr_neg = class_likelihood_ratios(y_true, y_pred)
        metrics_dict['Class Likelihood Ratios'] = {'LR+': lr_pos, 'LR-': lr_neg}

    metrics_dict['Precision Score'] = precision_score(y_true, y_pred)
    metrics_dict['Recall Score'] = recall_score(y_true, y_pred)

    # Balanced Accuracy
    metrics_dict['Balanced Accuracy'] = balanced_accuracy_score(y_true, y_pred)

    # Cohen's Kappa
    metrics_dict["Cohen's Kappa"] = cohen_kappa_score(y_true, y_pred)

    # Confusion Matrix
    metrics_dict['Confusion Matrix'] = confusion_matrix(y_true, y_pred)



    # Matthews Correlation Coefficient
    metrics_dict['Matthews Correlation Coefficient'] = matthews_corrcoef(y_true, y_pred)



    # Accuracy
    metrics_dict['Accuracy'] = accuracy_score(y_true, y_pred)

    # Classification Report
    class_rep = classification_report(y_true, y_pred, output_dict=True)
    metrics_dict['Classification Report'] = class_rep

    if "UP" in y_pred.values:
        # F1 ScorezcX
        f1 = f1_score(y_true, y_pred, pos_label='UP')
        metrics_dict['F1 Score'] = f1

        # F-beta Score
        beta = 1  # You can change the beta value as needed
        fbeta = fbeta_score(y_true, y_pred, beta=beta, pos_label='UP')
        metrics_dict[f'F-{beta} Score'] = fbeta
    else:
        f1 = f1_score(y_true, y_pred)
        metrics_dict['F1 Score'] = f1

        # F-beta Score
        beta = 1  # You can change the beta value as needed
        fbeta = fbeta_score(y_true, y_pred, beta=beta)
        metrics_dict[f'F-{beta} Score'] = fbeta

    # Hamming Loss (Multilabel Classification)
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        hamming_loss_val = hamming_loss(y_true, y_pred)
        metrics_dict['Hamming Loss'] = hamming_loss_val

    # Jaccard Score (Multilabel Classification)
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        jaccard = jaccard_score(y_true, y_pred, average='samples')
        metrics_dict['Jaccard Score'] = jaccard



    # Multilabel Confusion Matrix (Multilabel Classification)
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        multilabel_confusion = multilabel_confusion_matrix(y_true, y_pred)
        metrics_dict['Multilabel Confusion Matrix'] = multilabel_confusion.tolist()

    # Precision, Recall, F1 Score, Support (Multiclass Classification)
    if len(set(y_true)) > 2:
        prec_rec_f1_supp = precision_recall_fscore_support(y_true, y_pred, average=None)
        for i, label in enumerate(set(y_true)):
            metrics_dict[f'Precision (Class {label})'] = prec_rec_f1_supp[0][i]
            metrics_dict[f'Recall (Class {label})'] = prec_rec_f1_supp[1][i]
            metrics_dict[f'F1 Score (Class {label})'] = prec_rec_f1_supp[2][i]
            metrics_dict[f'Support (Class {label})'] = prec_rec_f1_supp[3][i]

    # Precision Score (Multiclass Classification)
    if len(set(y_true)) > 2:
        prec_score = precision_score(y_true, y_pred, average=None)
        for i, label in enumerate(set(y_true)):
            metrics_dict[f'Precision (Class {label})'] = prec_score[i]

    # Recall Score (Multiclass Classification)
    if len(set(y_true)) > 2:
        rec_score = recall_score(y_true, y_pred, average=None)
        for i, label in enumerate(set(y_true)):
            metrics_dict[f'Recall (Class {label})'] = rec_score[i]



    # Zero-One Loss (Multiclass Classification)
    if len(set(y_true)) > 2:
        zero_one_loss_val = zero_one_loss(y_true, y_pred)
        metrics_dict['Zero-One Loss'] = zero_one_loss_val

    return metrics_dict


def confusion_matrix_metric_to_dict(cm):
    tn, fp, fn, tp = cm.ravel()

    metrics_dict = {
        'True Positive (TP)': tp,
        'True Negative (TN)': tn,
        'False Positive (FP)': fp,
        'False Negative (FN)': fn
    }

    return metrics_dict


def flatten_dict(d, parent_key='', sep=' '):
    flattened = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            flattened.update(flatten_dict(v, new_key, sep=sep))
        else:
            flattened[new_key] = v
    return flattened


def convert_metrics_to_record(metrics_dict):
    metrics_dict['Confusion Matrix'] = confusion_matrix_metric_to_dict(metrics_dict['Confusion Matrix'])

    metrics_dict = flatten_dict(metrics_dict)
    return metrics_dict


def get_additional_metrics(y_true, y_pred):
    return AdditionalMetrics().binary_classification(y_true, y_pred)


def load_json(json_path):
    with open(json_path, 'r') as reader:
        data = json.load(reader)
    return data


def write_json(json_path, data):
    with open(json_path, "w") as write_file:
        json.dump(data, write_file)


def confusion_matrix_to_json_format(y_test, y_pred):
    temp = confusion_matrix(y_test, y_pred)

    matrix_data = []
    row_index = -1
    for row in temp:
        matrix_data.append([])
        row_index += 1
        for col in row:
            matrix_data[row_index].append(float(col))
    return matrix_data

def dataframe_column_barh_plot(dataframe, column_name, title, y_label):
    dataframe[column_name].plot.barh()
    plt.title(title)
    plt.xlabel(column_name)
    plt.ylabel(y_label)
    plt.show()


class ClassificationMetric:
    def __init__(self, y_test, y_pred):
        self.y_test = y_test
        self.y_pred = y_pred
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.precision = precision_score(self.y_test, self.y_pred)
        self.recall = recall_score(self.y_test, self.y_pred)
        self.confusion_matrix_to_list = confusion_matrix(self.y_test, self.y_pred)

    def details(self):
        print(f"accuracy score :{self.accuracy * 100:.2f}%")
        print(f"precision score :{self.precision * 100:.2f}%")
        print(f"recall score :{self.recall * 100:.2f}%")
        return self.accuracy, self.precision, self.recall

    def confusion_matrix_pie(self):
        metric_data = [value for row in self.confusion_matrix_to_list for value in row]
        plt.figure()
        plt.title("Confusion metric Pie")
        plt.pie(metric_data, autopct="%.2f%%", labels=[f'True Negative {metric_data[0]}',
                                                       f'False Negative {metric_data[1]}',
                                                       f'False Positive {metric_data[2]}',
                                                       f'True Positive {metric_data[3]}'
                                                       ])
        plt.show()

    def confusion_matrix_heatmap(self):
        ax = sns.heatmap(self.confusion_matrix_to_list, annot=True, cmap='Blues')

        ax.set_title('Confusion Matrix with labels')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Actual Values ')

        ax.xaxis.set_ticklabels(['False', 'True'])
        ax.yaxis.set_ticklabels(['False', 'True'])
        plt.show()


class RegressionMetric:
    def __init__(self, y_test, y_pred):
        self.y_test = y_test
        self.y_pred = y_pred
        self.mse = mean_squared_error(y_test, y_pred)
        self.r2 = r2_score(y_test, y_pred)

    def details(self):
        print(f"Mean squared error :{self.mse * 100:.2f}%")
        print(f"Coefficient of determination (R^2) :{self.r2 * 100:.2f}%")
        return self.mse, self.r2

    def curves(self):
        result_df = pd.DataFrame(data={'y_test': self.y_test, 'y_pred': self.y_pred})
        result_df = result_df.sort_values(by=['y_test', 'y_pred'])

        plt.figure()
        plt.plot(result_df['y_test'].to_list(), label='y_test')
        plt.plot(result_df['y_pred'].to_list(), label='y_pred')
        plt.legend()
        plt.show()


class MetricComparison:
    """
    To do List:
     - write data remove methods
    """
    sample = {"Regression metrics": {},
              "Classification metrics": {},
              "Ensemble metrics": {"Regression metrics": {},
                                   "Classification metrics": {}
                                   },
              "Deep learning metrics": {"Regression metrics": {},
                                        "Classification metrics": {}
                                        }
              }

    def __init__(self, json_path=None):
        self.data = load_json(json_path) if json_path else self.sample
        self.json_path = json_path

    def get_sample_json(self, path="./generated files/sample.json"):
        write_json(path, self.sample)

    def update_metric_data(self):
        write_json(self.json_path, self.data)

    # Regression add methods
    def add_regression_metric(self, regressor_name, regressor_mean_squared_error=None, regressor_r2_score=None,
                              **kwargs):
        self.data["Regression metrics"][regressor_name] = {"Mean squared error": regressor_mean_squared_error,
                                                           "Coefficient of determination (R^2)": regressor_r2_score,
                                                           **kwargs
                                                           }
        self.update_metric_data()

    def remove_regression_metric(self, regressor_name):
        removed_value = self.data["Regression metrics"].pop(regressor_name)
        print(f"Removed from data -> 'Regression metrics' -> {regressor_name}")
        print(f"value is {removed_value}")
        self.update_metric_data()

    def add_ensemble_regression_metric(self, regressor_name, regressor_mean_squared_error=None,
                                       regressor_r2_score=None, **kwargs):
        self.data["Ensemble metrics"]["Regression metrics"][regressor_name] = {
            "Mean squared error": regressor_mean_squared_error,
            "Coefficient of determination (R^2)": regressor_r2_score, **kwargs
        }
        self.update_metric_data()

    def add_deep_learning_regression_metric(self, regressor_name, regressor_mean_squared_error=None,
                                            regressor_r2_score=None, **kwargs):
        self.data["Deep learning metrics"]["Regression metrics"][regressor_name] = {
            "Mean squared error": regressor_mean_squared_error,
            "Coefficient of determination (R^2)": regressor_r2_score, **kwargs
        }
        self.update_metric_data()

    # Classification add methods
    def add_classification_metric(self, classification_name, classification_accuracy_score=None,
                                  classification_precision_score=None,
                                  classification_recall_score=None, classification_confusion_matrix=None, **kwargs):
        self.data["Classification metrics"][classification_name] = {"Accuracy score": classification_accuracy_score,
                                                                    "precision score": classification_precision_score,
                                                                    "recall score": classification_recall_score,
                                                                    "confusion matrix": classification_confusion_matrix,
                                                                    **kwargs
                                                                    }
        self.update_metric_data()

    def add_ensemble_classification_metric(self, classification_name, classification_accuracy_score=None,
                                           classification_precision_score=None,
                                           classification_recall_score=None, classification_confusion_matrix=None,
                                           **kwargs):
        self.data["Ensemble metrics"]["Classification metrics"][classification_name] = {
            "Accuracy score": classification_accuracy_score,
            "precision score": classification_precision_score,
            "recall score": classification_recall_score,
            "confusion matrix": classification_confusion_matrix, **kwargs
        }
        self.update_metric_data()

    def remove_ensemble_classification_metric(self, classification_name):
        removed_value = self.data["Ensemble metrics"]["Classification metrics"].pop(classification_name)
        print(f"Removed from data -> 'Ensemble metrics' -> 'Classification metrics' -> {classification_name}")
        print(f"value is {removed_value}")
        self.update_metric_data()

    def add_deep_learning_classification_metric(self, classification_name, classification_accuracy_score=None,
                                                classification_precision_score=None,
                                                classification_recall_score=None, classification_confusion_matrix=None,
                                                **kwargs):

        self.data["Deep learning metrics"]["Classification metrics"][classification_name] = {
            "Accuracy score": classification_accuracy_score,
            "precision score": classification_precision_score,
            "recall score": classification_recall_score,
            "confusion matrix": classification_confusion_matrix, **kwargs
        }
        self.update_metric_data()

    def choose_data_dict(self, type_of_technique=None):
        """

        Parameters
        ----------
        type_of_technique : str
            options available -['Regression metrics', 'Regression Ensemble metrics', 'Regression Deep learning metrics',
            'Classification metrics', 'Classification Ensemble metrics', 'Classification Deep learning metrics']

        Returns
        -------
        dict
            Point to data from the class self.data.

        """
        if type_of_technique == 'Regression metrics':
            data_dictionary = self.data['Regression metrics']
        elif type_of_technique == 'Regression Ensemble metrics':
            data_dictionary = self.data['Ensemble metrics']['Regression metrics']
        elif type_of_technique == 'Regression Deep learning metrics':
            data_dictionary = self.data['Deep learning metrics']['Regression metrics']
        elif type_of_technique == 'Classification metrics':
            data_dictionary = self.data['Classification metrics']
        elif type_of_technique == 'Classification Ensemble metrics':
            data_dictionary = self.data['Ensemble metrics']['Classification metrics']
        elif type_of_technique == 'Classification Deep learning metrics':
            data_dictionary = self.data['Deep learning metrics']['Classification metrics']
        else:
            raise NotImplementedError("This type of regression is not known, Please choose value from documentation")

        return data_dictionary

    def regression_models_metric_dataframe(self, type_of_technique='Regression metrics'):
        name_list = []
        mse_list = []
        r2_list = []

        for name, metric in self.choose_data_dict(type_of_technique).items():
            name_list.append(name)
            mse_list.append(metric['Mean squared error'])
            r2_list.append(metric['Coefficient of determination (R^2)'])

        metric_dataframe = pd.DataFrame({'Mean squared error': mse_list, 'Coefficient of determination (R^2)': r2_list},
                                        index=name_list)
        return metric_dataframe

    def classification_models_metric_dataframe(self, type_of_technique='Classification metrics'):
        name_list = []
        accuracy_list = []
        precision_list = []
        recall_list = []
        true_negative_list = []
        false_negative_list = []
        false_positive_list = []
        true_positive_list = []
        confusion_matrix_label_list = ['True Negative', 'False Positive', 'False Negative', 'True Positive']

        for name, metric in self.choose_data_dict(type_of_technique).items():
            name_list.append(name)
            accuracy_list.append(metric['Accuracy score'])
            precision_list.append(metric['precision score'])
            recall_list.append(metric['recall score'])
            tn, fp, fn, tp = np.array(metric['confusion matrix']).ravel()

            true_negative_list.append(tn)
            false_positive_list.append(fp)
            false_negative_list.append(fn)
            true_positive_list.append(tp)

        df_data = {'Accuracy score': accuracy_list, 'precision score': precision_list,
                   'recall score': recall_list, confusion_matrix_label_list[0]: true_negative_list,
                   confusion_matrix_label_list[1]: false_positive_list,
                   confusion_matrix_label_list[2]: false_negative_list,
                   confusion_matrix_label_list[3]: true_positive_list}

        metric_dataframe = pd.DataFrame(df_data, index=name_list)
        return metric_dataframe

    def techniques_metric_info(self, type_of_technique_list=None):
        all_implement_techniques = ['Regression metrics', 'Regression Ensemble metrics',
                                    'Regression Deep learning metrics', 'Classification metrics',
                                    'Classification Ensemble metrics', 'Classification Deep learning metrics']
        technique_list = type_of_technique_list if type_of_technique_list else all_implement_techniques
        for technique in technique_list:
            if technique in {'Regression metrics', 'Regression Ensemble metrics',
                             'Regression Deep learning metrics'}:
                df = self.regression_models_metric_dataframe(technique)
                print(df)
                print("*" * 50)
            else:
                df = self.classification_models_metric_dataframe(technique)
                print(df)
                print("*" * 50)

    def techniques_metric_dataframe(self, type_of_technique_list=None, sort_by_column_list=None):
        """

        Parameters
        ----------
        sort_by_column_list
        type_of_technique_list : str
            'Regression metrics', 'Regression Ensemble metrics', 'Regression Deep learning metrics',
             'Classification metrics', 'Classification Ensemble metrics', 'Classification Deep learning metrics'

        Returns
        -------

        """
        all_implement_techniques = ['Regression metrics', 'Regression Ensemble metrics',
                                    'Regression Deep learning metrics', 'Classification metrics',
                                    'Classification Ensemble metrics', 'Classification Deep learning metrics']
        technique_list = type_of_technique_list if type_of_technique_list else all_implement_techniques
        result = pd.DataFrame()
        for technique in technique_list:
            if technique in {'Regression metrics', 'Regression Ensemble metrics',
                             'Regression Deep learning metrics'}:
                df = self.regression_models_metric_dataframe(technique)
                result = pd.concat([result, df], sort=False)
            else:
                df = self.classification_models_metric_dataframe(technique)
                result = pd.concat([result, df], sort=False)

        return result.sort_values(by=sort_by_column_list) if sort_by_column_list else result

    def comparison_bar_plot(self, type_of_technique, column_name=None, all_columns_included=False, ascending=False,
                            title="auto",
                            y_label="Names of Techniques"):
        """

        Parameters
        ----------
        type_of_technique : str
            'Regression metrics', 'Regression Ensemble metrics', 'Regression Deep learning metrics',
             'Classification metrics', 'Classification Ensemble metrics', 'Classification Deep learning metrics'

        column_name
        all_columns_included
        ascending
        title
        y_label

        Returns
        -------

        """
        assert bool(column_name) or all_columns_included, "Please provide name of metric(column_name) or set " \
                                                          "all_columns_included to True"
        if type_of_technique in {'Regression metrics', 'Regression Ensemble metrics',
                                 'Regression Deep learning metrics'}:
            if column_name:
                df = self.regression_models_metric_dataframe(type_of_technique).sort_values(by=column_name,
                                                                                            ascending=ascending)
            else:
                df = self.regression_models_metric_dataframe(type_of_technique)
        else:
            if column_name:
                df = self.classification_models_metric_dataframe(type_of_technique).sort_values(by=column_name,
                                                                                            ascending=ascending)
            else:
                df = self.classification_models_metric_dataframe(type_of_technique)

        title_string = f"{type_of_technique} Comparison" if title == 'auto' else title

        if all_columns_included:
            for col in df.columns.to_list():
                dataframe_column_barh_plot(df, col, title_string, y_label)
        else:
            dataframe_column_barh_plot(df, column_name, title_string, y_label)

    def techniques_comparison_bar_plot(self, all_columns_included=True, regression_metric_column_name=None,
                                       classification_metric_column_name=None, type_of_technique_list=None,
                                       ascending=False, title="auto", y_label="Names of Techniques"):
        all_implement_techniques = ['Regression metrics', 'Regression Ensemble metrics',
                                    'Regression Deep learning metrics', 'Classification metrics',
                                    'Classification Ensemble metrics', 'Classification Deep learning metrics']
        technique_list = type_of_technique_list if type_of_technique_list else all_implement_techniques

        for technique in technique_list:
            if technique in {'Regression metrics', 'Regression Ensemble metrics', 'Regression Deep learning metrics'}:
                column_name = regression_metric_column_name
            else:
                column_name = classification_metric_column_name
            self.comparison_bar_plot(technique, column_name, all_columns_included, ascending, title, y_label)

# need to merge this function with above classes
# written in tabular-data-ai
