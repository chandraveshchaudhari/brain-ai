from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
from keras import layers
import joblib
from systematic_review import os_utils


def save_model(clf, model_file_name_with_pkl_extension):
    # save
    joblib.dump(clf, model_file_name_with_pkl_extension)


def load_model(model_file_path):

    if os_utils.get_file_extension_from_path(model_file_path) == "pkl":
        return joblib.load(model_file_path)
    if os_utils.get_file_extension_from_path(model_file_path) == "h5":
        return keras.models.load_model(model_file_path)


def fit_predict_and_save_model(clf, x_train, y_train, x_test, y_test, saved_model_file_name=None):
    clf.fit(x_train, y_train)
    # Make predictions using the testing set
    y_pred = clf.predict(x_test)
    if saved_model_file_name == "default_file_name":
        saved_model_file_name = f"{type(clf).__name__}.pkl"
        save_model(clf, saved_model_file_name)
    elif saved_model_file_name is not None:
        save_model(clf, saved_model_file_name)
    return {'Accuracy Score': accuracy_score(y_test, y_pred) * 100,
            'Precision Score': precision_score(y_test, y_pred) * 100,
            'Recall Score': recall_score(y_test, y_pred) * 100}


def get_sklearn_model_by_name(model_name):
    train_model_dict = {'LogisticRegression': LogisticRegression(max_iter=1000),
                        'SVC': svm.SVC(),
                        'KNeighborsClassifier': KNeighborsClassifier(),
                        'RandomForestClassifier': RandomForestClassifier(n_estimators=20),
                        }
    return train_model_dict[model_name]


def sklearn_model_train(model_name, x_train, y_train):
    clf = get_sklearn_model_by_name(model_name)
    return clf.fit(x_train, y_train)


# logistic regression
def get_logistic_regression(x_train, y_train, x_test, y_test, saved_model_file_name=None):
    clf = LogisticRegression(max_iter=1000)
    return fit_predict_and_save_model(clf, x_train, y_train, x_test, y_test, saved_model_file_name)


# svm.SVC
def get_svm_svc(x_train, y_train, x_test, y_test, saved_model_file_name=None):
    clf = svm.SVC()
    return fit_predict_and_save_model(clf, x_train, y_train, x_test, y_test, saved_model_file_name)


# kNeighbors classifier
def get_k_neighbors_classifier(x_train, y_train, x_test, y_test, saved_model_file_name=None):
    clf = KNeighborsClassifier()
    return fit_predict_and_save_model(clf, x_train, y_train, x_test, y_test, saved_model_file_name)


# random forest
def get_random_forest(x_train, y_train, x_test, y_test, saved_model_file_name=None):

    clf = RandomForestClassifier(n_estimators=20)
    return fit_predict_and_save_model(clf, x_train, y_train, x_test, y_test, saved_model_file_name)


# neural network
def get_neural_network(x_train, y_train, x_test, y_test, saved_model_file_name=None):
    model = train_neural_network(x_train, y_train)
    result = model.predict(x_test)

    if saved_model_file_name == "default_file_name":
        saved_model_file_name = f"Neural Network.h5"
        model.save(saved_model_file_name)
    elif saved_model_file_name is not None:
        model.save(saved_model_file_name)

    y_pred = []
    for i in result:
        if i[0] < 0.5:
            y_pred.append(0)
        else:
            y_pred.append(1)

    return {'Accuracy Score': accuracy_score(y_test, y_pred) * 100,
            'Precision Score': precision_score(y_test, y_pred) * 100,
            'Recall Score': recall_score(y_test, y_pred) * 100}


def train_neural_network(x_train, y_train):
    model = keras.Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10)
    return model
