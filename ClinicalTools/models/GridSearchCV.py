from sklearn import svm
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn import ensemble
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from ClinicalTools.utils.colors import bcolors
import numpy as np
import datetime


def evaluate_lr(x_train, y_train, parameters=None, cv=5):

    lr = LogisticRegressionCV(random_state=42, scoring='roc_auc', max_iter=100, n_jobs=-1, cv=10, )
    lr.fit(x_train, y_train)

    if parameters is None:
        parameters = {'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                      'C': lr.Cs_, # better tuned
                      'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                      'max_iter': [100, 200, 500, 1000]}
    model = LogisticRegression(class_weight='balanced',
                               random_state=42,)
    start = datetime.datetime.now().replace(microsecond=0)
    clf = GridSearchCV(model, parameters, scoring='roc_auc', cv=cv, n_jobs=-1)
    clf.fit(x_train, y_train)
    end = datetime.datetime.now().replace(microsecond=0)
    print(bcolors.FAIL + 'Logistic Regression' + bcolors.ENDC)
    print("Tuned hyper-parameters:(best parameters) ", clf.best_params_)
    print("Accuracy: {}".format(round(clf.best_score_, 3)))
    print('Consume Time: {}\n'.format(end-start))
    return clf.best_estimator_


def evaluate_svm(x_train, y_train, parameters=None, cv=5):
    if parameters is None:
        parameters = {'kernel': ['linear', 'rbf'],
                      'gamma': ['auto', 'scale'],
                      'degree': [3, 4],
                      'C': [0.01, 0.1, 1, 10]}
    model = svm.SVC(class_weight='balanced',
                    probability=True,
                    random_state=42,
                    max_iter=-1)
    start = datetime.datetime.now().replace(microsecond=0)
    clf = GridSearchCV(model, parameters, scoring='roc_auc', cv=cv, n_jobs=-1)
    clf.fit(x_train, y_train)
    end = datetime.datetime.now().replace(microsecond=0)
    print(bcolors.FAIL + 'Support Vector Classification' + bcolors.ENDC)
    print("Tuned hyper-parameters:(best parameters) ", clf.best_params_)
    print("Accuracy: {}".format(round(clf.best_score_, 3)))
    print('Consume Time: {}\n'.format(end-start))
    return clf.best_estimator_


def evaluate_gnb(x_train, y_train, parameters=None, cv=5):
    if parameters is None:
        parameters = {'var_smoothing': [1e-9*i for i in np.logspace(0, 10, 10)]}
    model = GaussianNB()
    start = datetime.datetime.now().replace(microsecond=0)
    clf = GridSearchCV(model, parameters, scoring='roc_auc', cv=cv, n_jobs=-1)
    clf.fit(x_train, y_train)
    end = datetime.datetime.now().replace(microsecond=0)
    print(bcolors.FAIL + 'Gaussian NB' + bcolors.ENDC)
    print("Tuned hyper-parameters:(best parameters) ", clf.best_params_)
    print("Accuracy: {}".format(round(clf.best_score_, 3)))
    print('Consume Time: {}\n'.format(end-start))
    return clf.best_estimator_


def evaluate_dt(x_train, y_train, parameters=None, cv=5):
    if parameters is None:
        parameters = {'criterion': ["gini", "entropy"],
                      'splitter': ['best', 'random'],
                      'max_depth': [3, 5, 10, 20],
                      'min_samples_split': [2, 3, 5],
                      'min_samples_leaf': [1, 2, 3],
                      'max_features': ["auto", "sqrt", "log2"]}
    model = tree.DecisionTreeClassifier(class_weight='balanced', random_state=42,)
    start = datetime.datetime.now().replace(microsecond=0)
    clf = GridSearchCV(model, parameters, scoring='roc_auc', cv=cv, n_jobs=-1)
    clf.fit(x_train, y_train)
    end = datetime.datetime.now().replace(microsecond=0)
    print(bcolors.FAIL + 'Decision Tree Classification' + bcolors.ENDC)
    print("Tuned hyper-parameters:(best parameters) ", clf.best_params_)
    print("Accuracy: {}".format(round(clf.best_score_, 3)))
    print('Consume Time: {}\n'.format(end-start))
    return clf.best_estimator_


def evaluate_rf(x_train, y_train, parameters=None, cv=5):
    if parameters is None:
        parameters = {'criterion': ["gini", "entropy"],
                      'max_depth': [3, 5, 10, 20],
                      'min_samples_split': [2, 3, 5],
                      'min_samples_leaf': [1, 2, 3],
                      'max_features': ["auto", "sqrt", "log2"],
                      'class_weight': ["balanced", "balanced_subsample"]}
    model = ensemble.RandomForestClassifier(random_state=42)
    start = datetime.datetime.now().replace(microsecond=0)
    clf = GridSearchCV(model, parameters, scoring='roc_auc', cv=cv, n_jobs=-1)
    clf.fit(x_train, y_train)
    end = datetime.datetime.now().replace(microsecond=0)
    print(bcolors.FAIL + 'Random Forest Classification' + bcolors.ENDC)
    print("Tuned hyper-parameters:(best parameters) ", clf.best_params_)
    print("Accuracy: {}".format(round(clf.best_score_, 3)))
    print('Consume Time: {}\n'.format(end-start))
    return clf.best_estimator_


def evaluate_gbm(x_train, y_train, parameters=None, cv=5):
    if parameters is None:
        parameters = {"loss": ["deviance"],
                      "learning_rate": [0.005, 0.05, 0.1, 1.0, 1.1, 1.2],
                      "min_samples_split": [2, 3],
                      "min_samples_leaf": [1, 2],
                      "max_depth": [3, 4, 5],
                      "max_features": ["auto"],
                      "criterion": ["friedman_mse"],
                      "n_estimators": [10, 40, 50, 60, 70, 100, 500]}
    model = ensemble.GradientBoostingClassifier(random_state=42)
    start = datetime.datetime.now().replace(microsecond=0)
    clf = GridSearchCV(model, parameters, scoring='roc_auc', cv=cv, n_jobs=-1)
    clf.fit(x_train, y_train)
    end = datetime.datetime.now().replace(microsecond=0)
    print(bcolors.FAIL + 'Gradient Boosting Decision Classification' + bcolors.ENDC)
    print("Tuned hyper-parameters:(best parameters) ", clf.best_params_)
    print("Accuracy: {}".format(round(clf.best_score_, 3)))
    print('Consume Time: {}\n'.format(end-start))
    return clf.best_estimator_


def evaluate_ada(x_train, y_train, parameters=None, cv=5):
    if parameters is None:
        parameters = {'learning_rate': [0.001, 0.005, 0.05, 0.1, 1.0],
                      'n_estimators': [10, 50, 100, 500, 600, 700, 1000],
                      'algorithm': ['SAMME', 'SAMME.R']}
    model = ensemble.AdaBoostClassifier(random_state=42)
    start = datetime.datetime.now().replace(microsecond=0)
    clf = GridSearchCV(model, parameters, scoring='roc_auc', cv=cv, n_jobs=-1)
    clf.fit(x_train, y_train)
    end = datetime.datetime.now().replace(microsecond=0)
    print(bcolors.FAIL + 'ADABoosting Classification' + bcolors.ENDC)
    print("Tuned hyper-parameters:(best parameters) ", clf.best_params_)
    print("Accuracy: {}".format(round(clf.best_score_, 3)))
    print('Consume Time: {}\n'.format(end-start))
    return clf.best_estimator_


def evaluate_mlp(x_train, y_train, parameters=None, cv=5):
    if parameters is None:
        parameters = {
            'hidden_layer_sizes': [(100,), (64,), (32,), (64, 32)],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha': [0.0001, 0.001, 0.01],
            'batch_size': [128],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'momentum': [0.9, 0.8]
        }
    model = MLPClassifier(random_state=0)
    start = datetime.datetime.now().replace(microsecond=0)
    clf = GridSearchCV(model, parameters, scoring='roc_auc', cv=cv, n_jobs=-1)
    clf.fit(x_train, y_train)
    end = datetime.datetime.now().replace(microsecond=0)
    print(bcolors.FAIL + 'MLP Classification' + bcolors.ENDC)
    print("Tuned hyper-parameters:(best parameters) ", clf.best_params_)
    print("Accuracy: {}".format(round(clf.best_score_, 3)))
    print('Consume Time: {}\n'.format(end - start))
    return clf.best_estimator_


def evaluate_lgb(x_train, y_train, parameters=None, cv=5):
    if parameters is None:
        parameters = {
            'max_depth': [3, 4, 6, 8],
            'num_leaves': [10, 20, 30, 40],
            'learning_rate': [0.001, 0.01, 0.1, 1],
            'num_iterations': [100, 200, 300, 500]
        }
    model = lgb.LGBMClassifier(objective='binary',
                               is_unbalance=True,
                               metric=['binary_logloss', 'auc'])
    start = datetime.datetime.now().replace(microsecond=0)
    clf = GridSearchCV(model, parameters, scoring='roc_auc', cv=cv, n_jobs=-1)
    clf.fit(x_train, y_train)
    end = datetime.datetime.now().replace(microsecond=0)
    print(bcolors.FAIL + 'LightGbm Classification' + bcolors.ENDC)
    print("Tuned hyper-parameters:(best parameters) ", clf.best_params_)
    print("Accuracy: {}".format(round(clf.best_score_, 3)))
    print('Consume Time: {}\n'.format(end-start))
    return clf.best_estimator_


def multi_best_models(x_train, y_train, cv=5):
    best_lr = evaluate_lr(x_train, y_train, cv=cv)
    best_svm = evaluate_svm(x_train, y_train, cv=cv)
    best_gnb = evaluate_gnb(x_train, y_train, cv=cv)
    # best_dt = evaluate_dt(x_train, y_train, cv=cv)
    best_rf = evaluate_rf(x_train, y_train, cv=cv)
    best_gbm = evaluate_gbm(x_train, y_train, cv=cv)
    best_ada = evaluate_ada(x_train, y_train, cv=cv)
    # best_lgb = evaluate_lgb(x_train, y_train, cv=cv)
    best_mlp = evaluate_mlp(x_train, y_train, cv=cv)

    return [best_lr, best_svm, best_gnb, best_rf, best_gbm, best_ada, best_mlp]