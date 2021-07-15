import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def impute_category_with_mode(x_train, x_test, classes=5):

    cate_vars = [n for n in x_train.columns if len(x_train[n].value_counts()) <= classes]

    # impute category vars with mode
    cate_imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    cate_imp.fit(x_train[cate_vars])

    x_train[cate_vars] = cate_imp.transform(x_train[cate_vars])
    x_test[cate_vars] = cate_imp.transform(x_test[cate_vars])

    return x_train, x_test


def impute_continue_with_mice(x_train, x_test):
    # impute continue vars with mice
    Iter_Imp = IterativeImputer(max_iter=10, random_state=42, n_nearest_features=10, min_value=0)
    Iter_Imp.fit(x_train)

    x_train = pd.DataFrame(Iter_Imp.transform(x_train), columns=x_train.columns)
    x_test = pd.DataFrame(Iter_Imp.transform(x_test), columns=x_test.columns)

    return x_train, x_test
