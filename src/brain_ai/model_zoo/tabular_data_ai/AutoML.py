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

        if type(data) is str:
            data = DataHandler(data).dataframe()
            if type(data) is not pd.DataFrame:
                raise TypeError("Data must be pandas DataFrame")

        if type(target_data_or_column_name) is str:
            self.target_column_name = target_data_or_column_name
            self.complete_data = data.copy()
            self.target_data = data[self.target_column_name]
            self.data_without_target = data.drop(columns=self.target_column_name)
        else:
            self.target_column_name = target_data_or_column_name.columns[0]
            self.data_without_target = data.copy()
            self.target_data = target_data_or_column_name
            self.complete_data = pd.concat([self.data_without_target, self.target_data], axis=1)

        if split_data_by_column_name_and_value_dict is not None and test_size is not None:
            raise ValueError("Both split_data_by_column_name_and_value_dict and test_size cannot be used together.")
        elif split_data_by_column_name_and_value_dict is None and test_size is None:
            raise ValueError("Either split_data_by_column_name_and_value_dict or test_size must be used."
                             "you can mention test_size as 0.33 for 33% test data.")

        if test_size is not None:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data_without_target,
                                                                                    self.target_data,
                                                                                    test_size=test_size,
                                                                                    random_state=42)
            self.training_data = pd.concat([self.x_train, self.y_train], axis=1)
            self.testing_data = pd.concat([self.x_test, self.y_test], axis=1)

        if split_data_by_column_name_and_value_dict is not None:
            self.training_data = self.complete_data.loc[
                self.complete_data[split_data_by_column_name_and_value_dict.keys()[0]] <
                split_data_by_column_name_and_value_dict.values()[0]]
            self.testing_data = self.complete_data.loc[
                self.complete_data[split_data_by_column_name_and_value_dict.keys()[0]] >=
                split_data_by_column_name_and_value_dict.values()[0]]
            self.y_train = self.training_data[target_data_or_column_name]
            self.x_train = self.training_data.drop(columns=target_data_or_column_name)
            self.y_test = self.testing_data[target_data_or_column_name]
            self.x_test = self.testing_data.drop(columns=target_data_or_column_name)

        self.tabular_directory = tabular_directory

        if os.path.exists(self.tabular_directory):
            print(f"TabularAutoML directory already exists at {self.tabular_directory}")

        self.saved_models_directory_path = os.path.join(self.tabular_directory, 'Tabular AutoML Saved Models')
        os.makedirs(self.saved_models_directory_path, exist_ok=True)

        self.tabular_log_directory_path = os.path.join(self.tabular_directory, 'Log')
        os.makedirs(self.tabular_log_directory_path, exist_ok=True)

        if logger is None:
            self.logger = Logger(log_project_name="Tabular AutoML", log_directory_path=self.tabular_log_directory_path)
        else:
            self.logger = logger
        self.logger.welcome_log("Tabular AutoML")

    def train(self, clean_data=False):
        save_data_object = SaveData(self.saved_models_directory_path)
        pickle_file_name = os.path.join(self.saved_models_directory_path, 'all_predictions_dictionary.pkl')

        if os.path.isfile(pickle_file_name):
            all_predictions_dictionary = load_from_pickle(pickle_file_name)
            print(all_predictions_dictionary)
        else:
            all_predictions_dictionary = dict()
        if clean_data:
            if 'AutoKeras' not in all_predictions_dictionary:
                all_predictions_dictionary['AutoKeras'] = self.autokeras_automl()
                self.logger.info(f"all_predictions_dictionary: {all_predictions_dictionary}")
                save_data_object.save(all_predictions_dictionary['AutoKeras'], 'AutoKeras')

            if 'TPOT' not in all_predictions_dictionary:
                all_predictions_dictionary['TPOT'] = self.tpot_automl()
                save_data_object.save(all_predictions_dictionary['TPOT'], 'TPOT')

        if 'AutoGluon' not in all_predictions_dictionary:
            all_predictions_dictionary['AutoGluon'] = self.autogluon_automl()
            self.logger.info(f"all_predictions_dictionary: {all_predictions_dictionary}")
            save_data_object.save(all_predictions_dictionary['AutoGluon'], 'AutoGluon')

        if 'AutoSklearn' not in all_predictions_dictionary:
            all_predictions_dictionary['AutoSklearn'] = self.autosklearn_automl()
            save_data_object.save(all_predictions_dictionary['AutoSklearn'], 'AutoSklearn')

        if 'PyCaret' not in all_predictions_dictionary:
            all_predictions_dictionary['PyCaret'] = self.pycaret_automl()
            save_data_object.save(all_predictions_dictionary['PyCaret'], 'PyCaret')

        if 'ml_jar_automl' not in all_predictions_dictionary:
            all_predictions_dictionary['ml_jar_automl'] = self.ml_jar_automl()
            save_data_object.save(all_predictions_dictionary['ml_jar_automl'], 'ml_jar_automl')

        if 'H2O' not in all_predictions_dictionary:
            all_predictions_dictionary['H2O'] = self.h2o_automl()
            self.logger.info(f"all_predictions_dictionary: {all_predictions_dictionary}")
            save_data_object.save(all_predictions_dictionary['H2O'], 'H2O')

        save_to_pickle(pickle_file_name, all_predictions_dictionary)

        return all_predictions_dictionary

    def autogluon_automl(self, enable_text_special_features=False,
                         enable_text_ngram_features=False,
                         enable_raw_text_features=False,
                         enable_vision_features=False):

        from autogluon.tabular import TabularPredictor
        from autogluon.features.generators import AutoMLPipelineFeatureGenerator

        package_name = 'AutoGluon Tabular'
        saved_model_location = os.path.join(self.saved_models_directory_path, f'{package_name} Saved Models')
        os.makedirs(saved_model_location, exist_ok=True)
        tabular_auto_ml_log_path = os.path.join(self.tabular_log_directory_path,
                                                f'{package_name} Logs')
        os.makedirs(tabular_auto_ml_log_path, exist_ok=True)

        file_tabular_auto_ml_log_path = os.path.join(tabular_auto_ml_log_path, f'{package_name}.log')

        custom_feature_generator = AutoMLPipelineFeatureGenerator(
            enable_text_special_features=enable_text_special_features,
            enable_text_ngram_features=enable_text_ngram_features,
            enable_raw_text_features=enable_raw_text_features,
            enable_vision_features=enable_vision_features)

        predictor = TabularPredictor(label=self.target_column_name, problem_type='binary', log_to_file=True,
                                     log_file_path=file_tabular_auto_ml_log_path,
                                     path=saved_model_location).fit(self.training_data, presets='best_quality',
                                                                    feature_generator=custom_feature_generator
                                                                    )
        predictor.leaderboard().to_csv(os.path.join(saved_model_location, 'leaderboard.csv'))
        y_pred = predictor.predict_multi(self.x_test)

        return y_pred

    def autokeras_automl(self, autokeras_epochs=100, autokeras_max_trials=10):
        package_name = 'Auto Keras Tabular'

        saved_model_location = os.path.join(self.saved_models_directory_path, f'{package_name} Saved Models')
        os.makedirs(saved_model_location, exist_ok=True)

        self.logger.info(f"{package_name} Models will be saved here: {saved_model_location}")

        import tensorflow as tf
        import autokeras as ak

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
        return y_pred

    def tpot_automl(self):
        package_name = 'TPOT Tabular'

        saved_model_location = os.path.join(self.saved_models_directory_path, f'{package_name} Saved Models')
        os.makedirs(saved_model_location, exist_ok=True)

        self.logger.info(f"{package_name} Models will be saved here: {saved_model_location}")

        from tpot import TPOTClassifier

        clf = TPOTClassifier(generations=5, population_size=50, verbosity=2)
        clf.fit(self.x_train, self.y_train)

        tpot_y_pred = clf.predict(self.x_test)
        clf.export(os.path.join(saved_model_location, 'pipeline.py'))

        nn_clf = TPOTClassifier(config_dict='TPOT NN', template='Selector-Transformer-PytorchLRClassifier',
                                verbosity=2, population_size=10, generations=10)
        nn_clf.fit(self.x_train, self.y_train)
        nn_tpot_y_pred = nn_clf.predict(self.x_test)
        nn_clf.export(os.path.join(saved_model_location, 'NN_pipeline.py'))
        return {'tpot_y_pred': tpot_y_pred, 'nn_tpot_y_pred': nn_tpot_y_pred}

    def autosklearn_automl(self, time_allotted_for_this_task=7200):
        package_name = 'AutoSklearn Tabular'

        saved_model_location = os.path.join(self.saved_models_directory_path, f'{package_name} Saved Models')
        os.makedirs(saved_model_location, exist_ok=True)

        self.logger.info(f"{package_name} Models will be saved here: {saved_model_location}")

        import autosklearn.classification

        clf = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=time_allotted_for_this_task,
                                                               tmp_folder=saved_model_location,
                                                               delete_tmp_folder_after_terminate=False, )
        clf.fit(self.x_train, self.y_train)
        y_pred = clf.predict(self.x_test)
        leader_board = clf.leaderboard()
        if leader_board is pd.DataFrame:
            leader_board.to_csv(os.path.join(saved_model_location, 'leaderboard.csv'))
        else:
            self.logger.info(f"Leaderboard: {leader_board}")

        ensemble_dict = clf.show_models()
        try:
            write_json_file(os.path.join(saved_model_location, 'ensemble_dict.json'), ensemble_dict)
        except Exception:
            self.logger.info(f"Failed to save the {ensemble_dict} model.")

        return y_pred

    def pycaret_automl(self):
        package_name = 'PyCaret Tabular'

        saved_model_location = os.path.join(self.saved_models_directory_path, f'{package_name} Saved Models')
        os.makedirs(saved_model_location, exist_ok=True)

        self.logger.info(f"{package_name} Models will be saved here: {saved_model_location}")

        import pycaret

        from pycaret.classification import ClassificationExperiment
        clf = ClassificationExperiment()
        clf.setup(self.training_data, target=self.target_column_name, session_id=123)
        best_model = clf.compare_models(n_select=16)
        self.logger.info(f"Models: {best_model}")

        y_pred_dictionary = dict()
        for model in best_model:
            predictions = clf.predict_model(model, data=self.x_test)
            predictions.to_csv(os.path.join(saved_model_location, f'{model}.csv'))
            y_pred_dictionary[model] = predictions['prediction_label']

        clf.save_model(best_model[0], os.path.join(saved_model_location, 'PyCaret Pipeline'))
        return y_pred_dictionary

    def ml_jar_automl(self, mljar_total_time_limit=7200):
        package_name = 'ML Jar Tabular'

        saved_model_location = os.path.join(self.saved_models_directory_path, f'{package_name} Saved Models')
        os.makedirs(saved_model_location, exist_ok=True)

        self.logger.info(f"{package_name} Models will be saved here: {saved_model_location}")

        # mljar-supervised package
        from supervised.automl import AutoML

        # train models with AutoML
        automl = AutoML(mode="Perform", total_time_limit=mljar_total_time_limit, ml_task='binary_classification',
                        golden_features=False, features_selection=False, results_path=saved_model_location)
        automl.fit(self.x_train, self.y_train)

        # compute the accuracy on test data
        predictions = automl.predict(self.x_test)
        predictions_with_probability = automl.predict_all(self.x_test)
        if predictions_with_probability is pd.DataFrame:
            predictions_with_probability.to_csv(os.path.join(saved_model_location, 'predictions_with_probability.csv'))

        return predictions

    def h2o_automl(self, h2o_max_runtime_secs=7200):
        package_name = 'H2O Tabular'

        saved_model_location = os.path.join(self.saved_models_directory_path, f'{package_name} Saved Models')
        os.makedirs(saved_model_location, exist_ok=True)

        self.logger.info(f"{package_name} Models will be saved here: {saved_model_location}")

        # Start the H2O cluster (locally)
        h2o.init()

        train_h2o = h2o.H2OFrame(self.training_data)
        x = train_h2o.columns
        y = self.target_column_name
        x.remove(y)

        # For binary classification, response should be a factor
        self.training_data[y] = train_h2o[y].asfactor()

        # Run AutoML for 20 base models
        aml = H2OAutoML(max_models=2, seed=1)
        aml.train(x=x, y=y, training_frame=train_h2o)

        # View the AutoML Leaderboard
        y_pred_dictionary = dict()
        lb = aml.leaderboard
        if lb is pd.DataFrame:
            lb.to_csv(os.path.join(saved_model_location, 'leaderboard.csv'))
        else:
            try:
                self.logger.info(f"Leaderboard: {lb}")
            except Exception:
                self.logger.info(f"Could not provide Leaderboard in log file.")

        for model_id in lb['model_id']:
            model = h2o.get_model(model_id)
            y_pred = model.predict(h2o.H2OFrame(self.x_test))
            h2o.save_model(model=model, path=saved_model_location, force=True)
            y_pred_dictionary[model_id] = y_pred

        return y_pred_dictionary

    def predict(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def evaluate(self):
        pass

    def compare(self):
        pass
