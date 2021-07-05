"""
模拟退火算法优化
"""
from scipy.optimize import dual_annealing
from .evaluate import auc_acc_sen_spe_f1_scores
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from .colors import bcolors
import datetime


class SaOpt:
    def __init__(self, model, x_train, x_test, y_train, y_test,
                 maxiter=50):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model
        self.maxiter = maxiter

    def score(self, model):
        self.model.fit(self.x_train, self.x_test)
        prediction = model.predict(self.x_test)
        probability = model.predict_proba(self.test)[:, 1]
        results = auc_acc_sen_spe_f1_scores(self.y_train, prediction, probability)
        return -(results['F1'].values[0])

    # opt
    def opt_lr(self, x):
        C, max_iter = x
        model = LogisticRegression(class_weight='balanced',
                                   C=C,
                                   max_iter=int(max_iter),
                                   random_state=42, )

        return self.score(model)

    def opt_svc(self, x):
        parameters = {'kernel': ['linear', 'rbf'],
                      'gamma': ['auto', 'scale'],
                      'degree': [3, 5, 10],
                      'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}

        model = svm.SVC(class_weight='balanced',
                        probability=True,
                        random_state=42,
                        max_iter=-1)

    def opt_dt(self, x):
        pass

    def opt_gnb(self, x):
        pass

    def opt_rf(self, x):
        pass

    def opt_gbm(self, x):
        lr, min_split, min_leaf, max_d, n_estimators = x
        gbm = GradientBoostingClassifier(learning_rate=lr,
                                         min_samples_split=int(min_split),
                                         min_samples_leaf=int(min_leaf),
                                         max_depth=max_d,
                                         n_estimators=n_estimators,
                                         random_state=42)

        return self.score(gbm)

    def opt_lgb(self, x):
        pass

    # ————————————————————————
    # evaluation
    def evaluate_lr(self):
        # 依次：C, max_iter
        lw = [1e-3, 100]  # min
        up = [10, 1000]  # max
        start = datetime.datetime.now().replace(microsecond=0)
        ret = dual_annealing(self.opt_gbm,  # 需要优化的function
                             bounds=list(zip(lw, up)),
                             maxiter=self.maxiter,  # 迭代次数
                             initial_temp=5.e4,
                             seed=1234)
        end = datetime.datetime.now().replace(microsecond=0)
        best_x, best_y = ret.x, ret.fun
        best_C = best_x[0]
        best_max_iter = int(best_x[1])

        best_model = LogisticRegression(class_weight='balanced',
                                        C=best_C,
                                        max_iter=int(best_max_iter),
                                        random_state=42, )

        print(bcolors.FAIL + 'Logistic Regression' + bcolors.ENDC)
        print('Best parameters: {}'.format(best_model.get_params()))
        print("Best F1 score: {}".format(round(best_y, 3)))
        print('Consume Time: {}\n'.format(end - start))
        return best_model

    def evaluate_gbm(self):
        # 依次：learning rate, min_samples_split, min_samples_leaf, max_depth, n_estimators
        lw = [1e-3, 2, 1, 2, 10]  # min
        up = [1.1, 5, 3, 6, 1000]  # max
        start = datetime.datetime.now().replace(microsecond=0)
        ret = dual_annealing(self.opt_gbm,  # 需要优化的function
                             bounds=list(zip(lw, up)),
                             maxiter=self.maxiter,  # 迭代次数
                             initial_temp=5.e4,
                             seed=1234)
        end = datetime.datetime.now().replace(microsecond=0)
        best_x, best_y = ret.x, ret.fun
        # self.best_parameters['GBM parameters'] = best_x
        best_lr = best_x[0]
        best_min_samples_split = int(best_x[1])
        best_min_samples_leaf = int(best_x[2])
        best_max_depth = int(best_x[3])
        best_n_estimators = int(best_x[4])

        best_model = GradientBoostingClassifier(learning_rate=best_lr,
                                                min_samples_split=best_min_samples_split,
                                                min_samples_leaf=best_min_samples_leaf,
                                                max_depth=best_max_depth,
                                                n_estimators=best_n_estimators)

        print(bcolors.FAIL + 'Gradient Boosting Decision Classification' + bcolors.ENDC)
        print('Best parameters: {}'.format(best_model.get_params()))
        print("Best F1 score: {}".format(round(best_y, 3)))
        print('Consume Time: {}\n'.format(end - start))
        return best_model
