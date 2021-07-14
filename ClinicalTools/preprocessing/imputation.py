import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def impute_with_mode_mice(x_train, x_test, classes=5):

    cate_vars = [n for n in x_train.columns if len(x_train[n].value_counts()) == classes]
    continue_vars = [n for n in x_train.columns if len(x_train[n].value_counts()) > classes]

    # impute category vars with mode
    cate_imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    cate_imp.fit(x_train[cate_vars])

    x_train[cate_vars] = cate_imp.transform(cate_vars)
    x_test[cate_vars] = cate_imp.transform(cate_vars)

    # impute continue vars with mice
    Iter_Imp = IterativeImputer(max_iter=10, random_state=42, n_nearest_features=10, min_value=0)
    Iter_Imp.fit(x_train[continue_vars])

    x_train[continue_vars] = Iter_Imp.transform(x_train)
    x_test[continue_vars] = Iter_Imp.transform(x_test)
    return x_train, x_test
