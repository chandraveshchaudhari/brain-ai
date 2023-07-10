"""
Remember to put distinct name of modules and they should not have same name functions and class inside
Try to use absolute import and reduce cyclic imports to avoid errors
if there are more than one modules then import like this:
from tabular_data_ai import sample_func
"""
from sklearn.model_selection import train_test_split
from tabular_data_ai.machine_learning_algorithm import *
import os
import sys

sys.path.append("/home/chandravesh/PhDWork/PycharmProjects/data-processing/src")
from data_processing import utils, file_paths


class TabularAIExecutor:
    def __init__(self, tabular_data, target):
        self.target = target
        self.tabular_data = tabular_data
        # X_train, X_test, y_train, y_test
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.tabular_data, self.target,
                                                                                test_size=0.33, random_state=42)
        self.metric = dict()
        self.model_dict = {'LogisticRegression': get_logistic_regression, 'SVC': get_svm_svc,
                           'KNeighborsClassifier': get_k_neighbors_classifier,
                           'RandomForestClassifier': get_random_forest,
                           'Neural Network': get_neural_network}

    # def compare_models_results(self):
    #     use prediction-techniques-comparison

    def add_model(self, model, model_name):
        self.model_dict[model_name] = model(self.x_train, self.y_train, self.x_test, self.y_test)

    def execute_all_models(self, save_models=False):
        # TODO: use save_model argument to put model path.
        # TODO: don't train model if it is already trained and saved.
        for model_name, model in self.model_dict.items():
            if save_models:
                self.metric[model_name] = model(self.x_train, self.y_train, self.x_test, self.y_test,
                                                "default_file_name")
            else:
                self.metric[model_name] = model(self.x_train, self.y_train, self.x_test, self.y_test)

        self.save()
        return self.metric

    def save(self, file_name="IT_Industry_Model_Metric.json"):
        utils.write_json_file(os.path.join(file_paths.FilePaths().TabularDataMetric_directory_path,
                                           file_name), self.metric)

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

    def inference(self, data_point_list):
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
            model = sklearn_model_train(model_name, self.x_train, self.y_train)
            return model.predict(data_point_list)
