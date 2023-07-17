"""
Remember to put distinct name of modules and they should not have same name functions and class inside
Try to use absolute import and reduce cyclic imports to avoid errors
if there are more than one modules then import like this:
from tabular_data_ai import sample_func
"""
import logging

import mlflow
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from brain_ai.data_processing.feature_engineering import FeatureEngineering
from brain_ai.data_processing.wrangling import DataClean
from brain_ai.model_zoo.tabular_data_ai.machine_learning_algorithm import get_logistic_regression, \
    get_k_neighbors_classifier, get_random_forest, get_neural_network, get_svm_svc, train_neural_network, \
    sklearn_model_train, KerasNeuralNetwork


class TabularAIExecutor:
    def __init__(self, tabular_data, target_column_name, test_size=0.33, tracking_uri=None, feature_engineering=True,
                 data_wrangling=True):
        # To-do: put following in configDataClean(self.feature_engineered_tabular_data,
        # target_column_name).execute(10000, 'percentage', 0)
        # Set the tracking URI
        self.tracking_uri = tracking_uri

        if feature_engineering and data_wrangling:
            self.feature_engineered_tabular_data = FeatureEngineering(tabular_data, target_column_name).scale()
            self.data_wrangled_list = DataClean(self.feature_engineered_tabular_data,
                                                        target_column_name).execute(10000, 'percentage', 0)
            self.total_data = pd.DataFrame.from_dict(self.data_wrangled_list, 'index')

            print(self.total_data)
            self.data_target = self.total_data.pop(target_column_name)
        elif feature_engineering:
            self.feature_engineered_tabular_data = FeatureEngineering(tabular_data, target_column_name).scale()
            total_data = self.feature_engineered_tabular_data
            data_target = self.feature_engineered_tabular_data.pop(target_column_name)
        elif data_wrangling:
            self.data_wrangled_tabular_data = DataClean(tabular_data, target_column_name).execute()
            total_data = self.data_wrangled_tabular_data
            data_target = self.data_wrangled_tabular_data.pop(target_column_name)
        else:
            total_data = tabular_data
            data_target = tabular_data.pop(target_column_name)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.total_data, self.data_target,
                                                                                test_size=test_size, random_state=42)
        self.metric = dict()
        # self.model_dict = {'LogisticRegression': get_logistic_regression, 'SVC': get_svm_svc,
        #                    'KNeighborsClassifier': get_k_neighbors_classifier,
        #                    'RandomForestClassifier': get_random_forest,
        #                    'Neural Network': get_neural_network}
        self.models = [LogisticRegression(max_iter=1000), svm.SVC(), KNeighborsClassifier(),
                       RandomForestClassifier(n_estimators=20), KerasNeuralNetwork()]

    # def compare_models_results(self):
    #     use prediction-techniques-comparison

    def train_and_test(self):
        for model in self.models:
            if self.tracking_uri is not None:
                mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.autolog()
            clf = model.fit(self.x_train, self.y_train)
            y_pred = clf.predict(self.x_test)
            self.metric[type(model).__name__] = (self.y_test, y_pred)
        return self.metric

    def add_model(self, model):
        self.models.append(model)

    # def execute_all_models(self, save_models=False, mlflow_log=True):
    #     if mlflow_log:
    #         mlflow.autolog()
    #
    #     # TODO: use save_model argument to put model path.
    #     # TODO: don't train model if it is already trained and saved.
    #     for model_name, model in self.model_dict.items():
    #         logging.info(f"Training {model_name} model")
    #         if save_models:
    #             self.metric[model_name] = model(self.x_train, self.y_train, self.x_test, self.y_test,
    #                                             "default_file_name")
    #         else:
    #             self.metric[model_name] = model(self.x_train, self.y_train, self.x_test, self.y_test)
    #
    #     self.save()
    #     return self.metric

    def save(self, file_name="IT_Industry_Model_Metric.json"):
        pass

    def collecting_results_into_applicable_format(self):
        # get applicable format from prediction-techniques-comparison
        pass

    def best_model(self):
        # find best model from prediction-techniques-comparison
        model_name = list(self.best_metric().keys())[0]
        print("best model: ", model_name)
        return self.model_dict[model_name]

    def best_metric(self):
        best_score = 0
        selected_metric = None

        for model_name, scores in self.metric.items():
            if scores['Accuracy Score'] > best_score:
                selected_metric = {model_name: scores}
                best_score = scores['Accuracy Score']

        return selected_metric

    def inference(self, data_point_list, mlflow_log=True):

        model_name = list(self.best_metric().keys())[0]
        print(f"Using {model_name} model for inference.")

        if model_name == 'Neural Network':
            model = train_neural_network(self.x_train, self.y_train)
            result = model.predict(data_point_list)
            y_pred = []
            for i in result:
                if i[0] < 0.5:
                    y_pred.append(0)
                else:
                    y_pred.append(1)
            return y_pred
        else:
            model = sklearn_model_train(model_name, self.x_train, self.y_train, mlflow_log)
            return model.predict(data_point_list)
