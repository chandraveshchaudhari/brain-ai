"""
Remember to put distinct name of modules and they should not have same name functions and class inside
Try to use absolute import and reduce cyclic imports to avoid errors
if there are more than one modules then import like this:
from tabular_data_ai import sample_func
"""
import logging
import os
import h2o
from h2o.automl import H2OAutoML

import mlflow
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from brain_ai.data_processing.feature_engineering import FeatureEngineering
from brain_ai.data_processing.wrangling import DataClean
from brain_ai.explainability.comparison import get_additional_metrics, calculate_all_classification_metrics, \
    convert_metrics_to_record
from brain_ai.model_zoo.tabular_data_ai.machine_learning_algorithm import get_logistic_regression, \
    get_k_neighbors_classifier, get_random_forest, get_neural_network, get_svm_svc, train_neural_network, \
    sklearn_model_train, KerasNeuralNetwork
from brain_ai.utilities.data_handling import DataHandler, write_json_file, SaveData, save_to_pickle, load_from_pickle
from brain_ai.utilities.log_handling import Logger


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


class TabularAutoML:
    def __init__(self, data, target_data_or_column_name, split_data_by_column_name_and_value_dict=None, test_size=None,
                 logger=None, tabular_directory=os.getcwd()):
        self.tabular_directory = tabular_directory

        if os.path.exists(self.tabular_directory):
            print(f"TabularAutoML directory already exists at {self.tabular_directory}")

        self.saved_models_directory_path = os.path.join(self.tabular_directory, 'Tabular AutoML Saved Models')
        print(f"Tabular AutoML Models will be saved here: {self.saved_models_directory_path}")
        os.makedirs(self.saved_models_directory_path, exist_ok=True)

        self.tabular_log_directory_path = os.path.join(self.tabular_directory, 'Log')
        print(f"Tabular AutoML Logs will be saved here: {self.tabular_log_directory_path}")
        os.makedirs(self.tabular_log_directory_path, exist_ok=True)

        if logger is None:
            self.logger = Logger(log_project_name="Tabular AutoML", log_directory_path=self.tabular_log_directory_path)
        else:
            self.logger = logger
        self.logger.welcome_log("Tabular AutoML")

        self.prediction_data_directory_path = os.path.join(self.tabular_directory, 'Prediction Data')
        os.makedirs(self.prediction_data_directory_path, exist_ok=True)
        self.prediction_dictionary_file_path = os.path.join(self.prediction_data_directory_path,
                                                            'prediction_dictionary.json')
        self.logger.info(f"Predicted Data will be saved here: {self.prediction_data_directory_path}")
        if os.path.exists(self.prediction_dictionary_file_path):
            self.logger.info(f"Prediction Data file already exists at {self.prediction_dictionary_file_path}")
            self.prediction_dictionary = DataHandler(self.prediction_dictionary_file_path).load()
            self.logger.info(f"Prediction Data file loaded from {self.prediction_dictionary_file_path} and contains"
                             f" {self.prediction_dictionary}.")
        else:
            self.logger.info(f"Prediction Data file not found at {self.prediction_dictionary_file_path}. Creating a "
                             f"new one.")
            self.prediction_dictionary = dict()

        if type(data) is str:
            self.logger.info(f"Loading data from {data}")
            data = DataHandler(data).dataframe()
            if type(data) is not pd.DataFrame:
                raise TypeError("Data must be pandas DataFrame")

        if type(target_data_or_column_name) is str:
            self.logger.info(f"target column is {target_data_or_column_name}")

            self.target_column_name = target_data_or_column_name
            self.complete_data = data.copy()
            self.target_data = data[self.target_column_name]
            self.data_without_target = data.drop(columns=self.target_column_name)
        else:
            self.logger.info(f"target data is provided directly: {len(target_data_or_column_name)}")

            self.target_column_name = target_data_or_column_name.columns[0]

            self.logger.info(f"target column is {self.target_column_name}")
            self.data_without_target = data.copy()
            self.target_data = target_data_or_column_name
            self.complete_data = pd.concat([self.data_without_target, self.target_data], axis=1)

        if split_data_by_column_name_and_value_dict is not None and test_size is not None:
            raise ValueError("Both split_data_by_column_name_and_value_dict and test_size cannot be used together.")
        elif split_data_by_column_name_and_value_dict is None and test_size is None:
            raise ValueError("Either split_data_by_column_name_and_value_dict or test_size must be used."
                             "you can mention test_size as 0.33 for 33% test data.")

        if test_size is not None:
            print(f"test_size is {test_size}")
            self.logger.info(f"test_size is {test_size}")
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data_without_target,
                                                                                    self.target_data,
                                                                                    test_size=test_size,
                                                                                    random_state=42)
            self.training_data = pd.concat([self.x_train, self.y_train], axis=1)
            self.testing_data = pd.concat([self.x_test, self.y_test], axis=1)

        if split_data_by_column_name_and_value_dict is not None:
            print(f"split_data_by_column_name_and_value_dict is {split_data_by_column_name_and_value_dict}")
            self.logger.info(f"split_data_by_column_name_and_value_dict is {split_data_by_column_name_and_value_dict}")

            self.training_data = self.complete_data.loc[
                self.complete_data[split_data_by_column_name_and_value_dict.keys()[0]] <
                split_data_by_column_name_and_value_dict.values()[0]]
            self.testing_data = self.complete_data.loc[
                self.complete_data[split_data_by_column_name_and_value_dict.keys()[0]] >=
                split_data_by_column_name_and_value_dict.values()[0]]
            self.logger.info(f"training data = data[{split_data_by_column_name_and_value_dict.keys()[0]}] < "
                             f"{split_data_by_column_name_and_value_dict.values()[0]}")
            self.logger.info(f"testing data = data[{split_data_by_column_name_and_value_dict.keys()[0]}] >= "
                             f"{split_data_by_column_name_and_value_dict.values()[0]}")

            self.y_train = self.training_data[target_data_or_column_name]
            self.x_train = self.training_data.drop(columns=target_data_or_column_name)
            self.y_test = self.testing_data[target_data_or_column_name]
            self.x_test = self.testing_data.drop(columns=target_data_or_column_name)

    def train_predict(self, clean_data=False):

        if clean_data:
            print("clean data flag is True!")
            self.logger.info("the data is cleaned so following models will also be trained: 'Auto Keras Tabular' and "
                             "'TPOT Tabular'")

            if 'Auto Keras Tabular' in self.prediction_dictionary:
                self.logger.info("Auto Keras Tabular is already trained. Skipping...")
            else:
                print("training Auto Keras Tabular")
                self.autokeras_automl()

            if 'TPOT Tabular' in self.prediction_dictionary:
                self.logger.info("TPOT Tabular is already trained. Skipping...")
            else:
                print("training TPOT Tabular")
                self.tpot_automl()

        if 'AutoGluon Tabular' in self.prediction_dictionary:
            self.logger.info("AutoGluon Tabular is already trained. Skipping...")
        else:
            print("training AutoGluon Tabular")
            self.autogluon_automl()

        if 'AutoSklearn Tabular' in self.prediction_dictionary:
            self.logger.info("AutoSklearn Tabular is already trained. Skipping...")
        else:
            print("training AutoSklearn Tabular")
            self.autosklearn_automl()

        if 'PyCaret Tabular' in self.prediction_dictionary:
            self.logger.info("PyCaret Tabular is already trained. Skipping...")
        else:
            print("training PyCaret Tabular")
            self.pycaret_automl()

        if 'ML Jar Tabular' in self.prediction_dictionary:
            self.logger.info("ML Jar Tabular is already trained. Skipping...")
        else:
            print("training ML Jar Tabular")
            self.ml_jar_automl()

        if 'H2O Tabular' in self.prediction_dictionary:
            self.logger.info("H2O Tabular is already trained. Skipping...")
        else:
            print("training H2O Tabular")
            self.h2o_automl()

        self.logger.info(f"Trained models are saved at {self.saved_models_directory_path} and "
                         f"predictions are saved at {self.prediction_data_directory_path}.")
        return True

    def autogluon_automl(self, enable_text_special_features=False,
                         enable_text_ngram_features=False,
                         enable_raw_text_features=False,
                         enable_vision_features=False):

        from autogluon.tabular import TabularPredictor
        from autogluon.features.generators import AutoMLPipelineFeatureGenerator
        self.logger.welcome_log("AutoGluon Tabular")

        package_name = 'AutoGluon Tabular'
        saved_model_location = os.path.join(self.saved_models_directory_path, f'{package_name} Saved Models')
        os.makedirs(saved_model_location, exist_ok=True)
        tabular_auto_ml_log_path = os.path.join(self.tabular_log_directory_path,
                                                f'{package_name} Logs')
        os.makedirs(tabular_auto_ml_log_path, exist_ok=True)

        self.logger.info(f"{package_name} Models will be saved here: {saved_model_location}"
                         f" and logs will be saved here: {tabular_auto_ml_log_path}")
        file_tabular_auto_ml_log_path = os.path.join(tabular_auto_ml_log_path, f'{package_name}.log')

        custom_feature_generator = AutoMLPipelineFeatureGenerator(
            enable_text_special_features=enable_text_special_features,
            enable_text_ngram_features=enable_text_ngram_features,
            enable_raw_text_features=enable_raw_text_features,
            enable_vision_features=enable_vision_features)

        self.logger.info(f"TabularPredictor(label={self.target_column_name}, problem_type='binary', log_to_file=True,"
                         f"log_file_path={file_tabular_auto_ml_log_path},"
                         f"path={saved_model_location}).fit(self.training_data, presets='best_quality',"
                         "feature_generator=AutoMLPipelineFeatureGenerator("
                         f"enable_text_special_features={enable_text_special_features},"
                         f"enable_text_ngram_features={enable_text_ngram_features},"
                         f"enable_raw_text_features={enable_raw_text_features},"
                         f"enable_vision_features={enable_vision_features}))")

        predictor = TabularPredictor(label=self.target_column_name, problem_type='binary', log_to_file=True,
                                     log_file_path=file_tabular_auto_ml_log_path,
                                     path=saved_model_location).fit(self.training_data, presets='best_quality',
                                                                    feature_generator=custom_feature_generator
                                                                    )

        predictor.leaderboard().to_csv(os.path.join(saved_model_location, 'leaderboard.csv'))
        self.logger.info(f"Leaderboard of {package_name} is saved at {saved_model_location}.")

        y_pred = predictor.predict_multi(self.x_test)

        for model_name, predictions in y_pred.items():
            self.save_details(package_name, model_name, predictions)

        return y_pred

    def autokeras_automl(self, autokeras_epochs=100, autokeras_max_trials=10):
        package_name = 'Auto Keras Tabular'
        self.logger.welcome_log(package_name)

        saved_model_location = os.path.join(self.saved_models_directory_path, f'{package_name} Saved Models')
        os.makedirs(saved_model_location, exist_ok=True)

        self.logger.info(f"{package_name} Models will be saved here: {saved_model_location}")

        import tensorflow as tf
        import autokeras as ak
        self.logger.info(
            f"clf = ak.StructuredDataClassifier(max_trials={autokeras_max_trials}, directory={saved_model_location})")
        # Initialize the structured data classifier.
        clf = ak.StructuredDataClassifier(max_trials=autokeras_max_trials, directory=saved_model_location
                                          )  # It tries 3 different models.
        # Feed the structured data classifier with training data.
        self.logger.info(
            f"Training the {package_name} model using {autokeras_epochs} epochs with {autokeras_max_trials} trials.")
        clf.fit(
            # The path to the train.csv file.
            x=self.x_train,
            # The name of the label column.
            y=self.y_train,
            epochs=autokeras_epochs,
        )
        # Export as a Keras Model.
        model = clf.export_model()
        try:
            print(model.summary())
            self.logger.info(model.summary())
        except Exception:
            self.logger.error("Failed to get summary of the model.")
        self.logger.info(f"Exported the {package_name} model of type {type(model)}.")

        try:
            model.save("model_autokeras", save_format="tf")
            self.logger.info(f"Saved the {package_name} model using 'tf' save_format to {saved_model_location}.")
        except Exception:
            try:
                model.save("model_autokeras.h5")
                self.logger.info(f"Saved the {package_name} model using 'h5' save_format to {saved_model_location}.")
            except Exception:
                self.logger.error(f"Failed to save the {package_name} model.")

        y_pred = clf.predict(self.x_test)
        y_pred_df = pd.DataFrame(y_pred)

        self.save_details(package_name, 'AutoKeras Model', y_pred_df)

        return y_pred

    def tpot_automl(self):
        package_name = 'TPOT Tabular'
        self.logger.welcome_log(package_name)

        saved_model_location = os.path.join(self.saved_models_directory_path, f'{package_name} Saved Models')
        os.makedirs(saved_model_location, exist_ok=True)

        self.logger.info(f"{package_name} Models will be saved here: {saved_model_location}")

        from tpot import TPOTClassifier

        self.logger.info("clf = TPOTClassifier(generations=5, population_size=50, verbosity=2)")
        clf = TPOTClassifier(generations=5, population_size=50, verbosity=2)
        clf.fit(self.x_train, self.y_train)

        tpot_y_pred = clf.predict(self.x_test)
        tpot_y_pred_df = pd.DataFrame(tpot_y_pred)

        self.save_details(package_name, 'tpot_y_pred', tpot_y_pred_df)
        clf.export(os.path.join(saved_model_location, 'pipeline.py'))

        self.logger.info("nn_clf = TPOTClassifier(config_dict='TPOT NN', "
                         "template='Selector-Transformer-PytorchLRClassifier',"
                         "verbosity=2, population_size=10, generations=10)")
        nn_clf = TPOTClassifier(config_dict='TPOT NN', template='Selector-Transformer-PytorchLRClassifier',
                                verbosity=2, population_size=10, generations=10)
        nn_clf.fit(self.x_train, self.y_train)
        nn_tpot_y_pred = nn_clf.predict(self.x_test)
        nn_tpot_y_pred_df = pd.DataFrame(nn_tpot_y_pred)
        nn_clf.export(os.path.join(saved_model_location, 'NN_pipeline.py'))

        self.save_details(package_name, 'nn_tpot_y_pred', nn_tpot_y_pred_df)
        return {'tpot_y_pred': tpot_y_pred, 'nn_tpot_y_pred': nn_tpot_y_pred}

    def autosklearn_automl(self, time_allotted_for_this_task=7200):
        package_name = 'AutoSklearn Tabular'
        self.logger.welcome_log(package_name)

        saved_model_location = os.path.join(self.saved_models_directory_path, f'{package_name} Saved Models')

        self.logger.info(f"{package_name} Models will be saved here: {saved_model_location}")

        import autosklearn.classification

        self.logger.info(f"clf = autosklearn.classification.AutoSklearnClassifier("
                         f"time_left_for_this_task={time_allotted_for_this_task},"
                         f"tmp_folder={saved_model_location},"
                         f"delete_tmp_folder_after_terminate=False, )")
        clf = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=time_allotted_for_this_task,
                                                               tmp_folder=saved_model_location,
                                                               delete_tmp_folder_after_terminate=False, )
        clf.fit(self.x_train, self.y_train)
        y_pred = clf.predict(self.x_test)
        y_pred_df = pd.DataFrame(y_pred)
        leader_board = clf.leaderboard()
        if leader_board is pd.DataFrame:
            leader_board.to_csv(os.path.join(saved_model_location, 'leaderboard.csv'))
            self.logger.info(f"Leaderboard of {package_name} is saved at {saved_model_location}.")
        else:
            self.logger.info(f"Leaderboard of {package_name} is not saved as it is not a DataFrame.")
            self.logger.info(f"Leaderboard: {leader_board}")

        ensemble_dict = clf.show_models()

        try:
            self.logger.info(f"Ensemble Dict: {ensemble_dict}"
                             f"Saving the {package_name} model to {saved_model_location}.")
            write_json_file(os.path.join(saved_model_location, 'ensemble_dict.json'), ensemble_dict)
        except Exception:
            self.logger.info(f"Failed to save the {ensemble_dict} model.")

        self.save_details(package_name, 'Auto sklearn model', y_pred_df)
        return y_pred

    def pycaret_automl(self):
        package_name = 'PyCaret Tabular'
        self.logger.welcome_log(package_name)

        saved_model_location = os.path.join(self.saved_models_directory_path, f'{package_name} Saved Models')
        os.makedirs(saved_model_location, exist_ok=True)

        self.logger.info(f"{package_name} Models will be saved here: {saved_model_location}")

        from pycaret.classification import ClassificationExperiment

        self.logger.info(f"clf = ClassificationExperiment()"
                         f"clf.setup(self.training_data, target=self.target_column_name, session_id=123)")
        clf = ClassificationExperiment()
        clf.setup(self.training_data, target=self.target_column_name, session_id=123)
        best_model = clf.compare_models(n_select=16)
        self.logger.info(f"Models: {type(best_model).__name__}")

        y_pred_dictionary = dict()
        for model in best_model:
            predictions = clf.predict_model(model, data=self.x_test)
            model_name = type(model).__name__
            y_pred_df = predictions[['prediction_label']]
            self.save_details(package_name, model_name, y_pred_df)

        clf.save_model(best_model[0], os.path.join(saved_model_location, 'PyCaret Pipeline'))
        return y_pred_dictionary

    def ml_jar_automl(self, mljar_total_time_limit=7200):
        package_name = 'ML Jar Tabular'
        self.logger.welcome_log(package_name)

        saved_model_location = os.path.join(self.saved_models_directory_path, f'{package_name} Saved Models')
        os.makedirs(saved_model_location, exist_ok=True)

        self.logger.info(f"{package_name} Models will be saved here: {saved_model_location}")

        # mljar-supervised package
        from supervised.automl import AutoML

        self.logger.info(f"automl = AutoML(mode='Perform', total_time_limit={mljar_total_time_limit},"
                         f"ml_task='binary_classification', golden_features=False, features_selection=False,"
                         f"results_path={saved_model_location})")
        # train models with AutoML
        automl = AutoML(mode="Perform", total_time_limit=mljar_total_time_limit, ml_task='binary_classification',
                        golden_features=False, features_selection=False, results_path=saved_model_location)

        automl.fit(self.x_train, self.y_train)

        # compute the accuracy on test data
        predictions = automl.predict(self.x_test)
        predictions_df = pd.DataFrame(predictions)

        predictions_with_probability = automl.predict_all(self.x_test)
        self.logger.info(f"Predictions with probability: {type(predictions_with_probability)}")
        if predictions_with_probability is pd.DataFrame:
            self.logger.info(f"Saving predictions with probability at {saved_model_location}.")
            predictions_with_probability.to_csv(os.path.join(saved_model_location, 'predictions_with_probability.csv'))

        self.save_details(package_name, 'ML Jar Model', predictions_df)

        return predictions

    def h2o_automl(self):
        package_name = 'H2O Tabular'
        self.logger.welcome_log(package_name)

        saved_model_location = os.path.join(self.saved_models_directory_path, f'{package_name} Saved Models')
        os.makedirs(saved_model_location, exist_ok=True)

        self.logger.info(f"{package_name} Models will be saved here: {saved_model_location}")

        # Start the H2O cluster (locally)
        h2o.init()

        train_h2o = h2o.H2OFrame(self.training_data)
        y = self.target_column_name
        # For binary classification, response should be a factor
        train_h2o[y] = train_h2o[y].asfactor()

        x = train_h2o.columns
        x.remove(y)

        self.logger.info(f"aml = H2OAutoML(seed=1)"
                         f"aml.train(x=x, y=y, training_frame=train_h2o)")

        # Run AutoML for 20 base models
        aml = H2OAutoML(seed=1)
        aml.train(x=x, y=y, training_frame=train_h2o)

        # View the AutoML Leaderboard
        lb = aml.leaderboard
        if lb is pd.DataFrame:
            lb.to_csv(os.path.join(saved_model_location, 'leaderboard.csv'))
        else:
            try:
                self.logger.info(f"Leaderboard: {lb}")
            except Exception:
                self.logger.info(f"Could not provide Leaderboard in log file.")

        model_ids = lb['model_id'].as_data_frame()['model_id'].tolist()

        for model_id in model_ids:
            model = h2o.get_model(str(model_id))
            y_pred = model.predict(h2o.H2OFrame(self.x_test))
            y_pred_df = y_pred.as_data_frame()

            h2o.save_model(model=model, path=saved_model_location, force=True)
            self.save_details(package_name, model_id, y_pred_df)

        return

    def predict(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def evaluate(self):
        pass

    def performance_metrics(self):
        record_list = []
        for model_name, path in self.prediction_dictionary.items():
            if path is True:
                continue
            y_pred = pd.read_csv(path, low_memory=False)
            metric_generator = calculate_all_classification_metrics(self.y_test, y_pred)
            metric_generator_record = convert_metrics_to_record(metric_generator)
            model_metrics = {'model_name': model_name, **metric_generator_record}
            record_list.append(model_metrics)

        return pd.DataFrame(record_list)

    def save_performance_metrics(self, path=None):
        if path:
            self.performance_metrics().to_csv(path)
        else:
            self.performance_metrics().to_csv(os.path.join(self.saved_models_directory_path,
                                                           'performance_metrics.csv'))

    def save_details(self, automl_name, model_name, predictions):
        prediction_path = os.path.join(self.prediction_data_directory_path, f'{model_name}.csv')
        self.prediction_dictionary[automl_name] = True
        self.prediction_dictionary[model_name] = prediction_path
        predictions.to_csv(prediction_path, index=False)
        self.logger.info(f"Saved {model_name} predictions of {automl_name} at {prediction_path}.")
        write_json_file(self.prediction_dictionary_file_path,
                        self.prediction_dictionary)
        self.logger.info(f"Saved prediction dictionary with {model_name} predictions"
                         f" at {self.prediction_dictionary_file_path}.")
