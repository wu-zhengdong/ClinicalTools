import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import numpy as np
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


def plot_roc(true_y, pred_y_prob, savefile=None):

    csfont = {
        'size': 25,
        'family': 'Times New Roman',
        'weight': 'bold',
    }
    plt.rcParams['axes.labelsize'] = 15
    plt.rcParams['axes.labelweight'] = 'bold'

    auc = metrics.roc_auc_score(true_y, pred_y_prob)
    fpr, tpr, thresholds = metrics.roc_curve(true_y, pred_y_prob, pos_label=1)
    plt.style.use('seaborn-ticks')
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)
    ax.tick_params(colors='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC area = %0.3f)' % auc)

    ax.set_xlabel('False Positive Rate', fontdict=csfont)
    ax.set_ylabel('True Positive Rate', fontdict=csfont)
    ax.set_title('ROC Curve', fontdict=csfont)
    ax.legend(fontsize=20)
    plt.tight_layout()

    if savefile:
        plt.savefig(savefile)
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          savefile=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    csfont = {
        'size': 25,
        'family': 'Times New Roman',
        'weight': 'bold',
    }
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontdict=csfont)
    plt.xlabel('Predicted label', fontdict=csfont)
    if savefile:
        plt.savefig(savefile)
    plt.show()


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