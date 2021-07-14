from tqdm import tqdm
from sklearn import metrics
import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")


def auc_acc_sen_spe_f1_scores(true_y, pred_y_label, pred_y_prob):
    """
    Input:
        true_y: Ground true label
        pred_y_label: predicted label
        pred_y_prob: predicted probability. 1d array
    Output:
        auc score, accuracy, sensitivity, specificity
    """
    tn, fp, fn, tp = metrics.confusion_matrix(true_y, pred_y_label).ravel()
    spe = tn/(tn+fp)
    sen = tp/(tp+fn)
    acc = (tn + tp)/true_y.shape[0]
    auc = metrics.roc_auc_score(true_y, pred_y_prob)
    f1 = metrics.f1_score(y_true=true_y, y_pred=pred_y_label)
    result = pd.DataFrame({'AUC': auc, 'Accuracy': acc, 'F1': f1, 'Sensitivity': sen, 'Specificity': spe}, index=[0])
    return result


def get_ci(score, percentile, precision):
    """
    Input:
        score: a list or an array of scores
        percentile: list indicating percentile position; e.g: [0.025,.5,.975]
        precision: floating point
    Output:
        String: median(lower bound, upper bound);
    """
    score_df = pd.DataFrame(score)
    st = score_df.describe(percentile)
    lower = np.round(st.iloc[4, 0], precision)
    median = np.round(st.iloc[5, 0], precision)
    upper = np.round(st.iloc[6, 0], precision)
    ci = (str(median) + '(' + str(lower) + ', ' + str(upper) + ')')
    return ci


def bootstrap_scores(true_y, pred_y_label, pred_y_prob,
                     percentile=[0.025, 0.5, 0.975], precision=3, n_samples=1000):
    """
    Input:
        true_y: Ground true label
        pred_y_label: predicted label
        pred_y_prob: predicted probability. 2d array
        n_samples
    Output:
        boostrapped scores (auc, acc, sen, spe)
    """
    random.seed(0)

    n_bootstraps = n_samples
    bootstrapped_aucs = []
    bootstrapped_accs = []
    bootstrapped_spes = []
    bootstrapped_sens = []
    bootstrapped_f1s = []

    for i in tqdm(range(n_bootstraps)):
        indices = np.random.random_integers(0, len(true_y) - 1, len(true_y))  # random generate indices.
        shuffle_true_y = true_y[indices]
        shuffle_pred_y_label = pred_y_label[indices]
        shuffle_pred_y_prob = pred_y_prob[indices]
        if len(np.unique(shuffle_true_y)) < 2:
            continue
        score_df = auc_acc_sen_spe_f1_scores(shuffle_true_y, shuffle_pred_y_label, shuffle_pred_y_prob)

        bootstrapped_aucs.append(score_df['AUC'])
        bootstrapped_accs.append(score_df['Accuracy'])
        bootstrapped_f1s.append(score_df['F1'])
        bootstrapped_sens.append(score_df['Sensitivity'])
        bootstrapped_spes.append(score_df['Specificity'])

    auc_ci = get_ci(bootstrapped_aucs, percentile, precision)
    acc_ci = get_ci(bootstrapped_accs, percentile, precision)
    f1_ci = get_ci(bootstrapped_f1s, percentile, precision)
    sen_ci = get_ci(bootstrapped_sens, percentile, precision)
    spe_ci = get_ci(bootstrapped_spes, percentile, precision)

    ci_df = pd.DataFrame({'AUC': auc_ci,
                          'Accuracy': acc_ci,
                          'Sensitivity': sen_ci,
                          'Specificity': spe_ci,
                          'F1': f1_ci},
                         index=['Mdian(CI)'])

    return ci_df


def bootstrap(x_test, y_test, models,
              index_names=['LR', 'SVC', 'GNB', 'RF', 'GBM', 'ADA', 'MLP'],
              precision=3, random_state=42):

    assert len(models) == len(index_names), "The models and index_names's length must match!"

    np.random.seed(random_state)
    bootstrap_df = pd.DataFrame([])
    for model in models:
        prediction = model.predict(x_test)
        probability = model.predict_proba(x_test)[:, 1]
        scores = bootstrap_scores(y_test.values, prediction, probability,
                                  n_samples=1000, precision=precision, percentile=[0.025, 0.5, 0.975])
        bootstrap_df = pd.concat([bootstrap_df, scores], axis=0)
    bootstrap_df.index = index_names
    return bootstrap_df