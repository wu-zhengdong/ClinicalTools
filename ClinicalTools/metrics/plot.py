import os
import numpy as np
from ClinicalTools.utils.colors import ColorCode
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.calibration import calibration_curve

import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.default'] = 'regular'
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.labelsize'] = 15
plt.style.use('seaborn-ticks')


np.random.seed(0)


def plot_roc_curve(models, x_test, y_test,
                   colors=None, model_names=None,
                   save_path='./results/pics'):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if colors is None:
        '''see this link: https://matplotlib.org/stable/gallery/color/named_colors.html'''
        colors = ['blue', 'black', 'tan', 'darkgray', 'red', 'green', 'darkred']
    if model_names is None:
        model_names = ['LR', 'SVC', 'GNB', 'RF', 'GBM', 'ADA', 'MLP']

    csfont = {
        'size': 25,
        'family': 'Times New Roman',
        'weight': 'bold',
    }

    fig, ax = plt.subplots(figsize=(10, 10))

    for model, color, n in zip(models, colors, model_names):
        prob = model.predict_proba(x_test)[:, 1]
        # auc = metrics.roc_auc_score(y_test, prob)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, prob, pos_label=1)
        plt.plot(fpr, tpr, color=ColorCode[color], lw=2, label='{}'.format(n))

    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)
    ax.tick_params(colors='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax.set_xlabel('False Positive Rate', fontdict=csfont)
    ax.set_ylabel('True Positive Rate', fontdict=csfont)
    ax.set_title('ROC Curve', fontdict=csfont)
    ax.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'roc.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_cm(model, x_test, y_test, class_names=['0', '1'],
            cmap=plt.cm.Blues, save_path='./results/pics'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(model, x_test, y_test,
                                     display_labels=class_names,
                                     cmap=cmap,
                                     normalize=normalize)
        disp.ax_.set_title(title)
        plt.savefig(os.path.join(save_path, '{}.png'.format(title)), dpi=300)

    plt.show()


def plot_calibration_curve(calibration_model, x_train, y_train,
                           x_test, y_test, name='model', save_path='./pics',):
    '''
    Usually, we only use the model with best performance.
    '''
    plt.figure(figsize=(10, 10))
    ax = plt.subplot2grid((1, 1), (0, 0), rowspan=2)

    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    calibration_model.fit(x_train, y_train)
    if hasattr(calibration_model, "predict_proba"):
        prob_pos = calibration_model.predict_proba(x_test)[:, 1]
    else:  # use decision function
        prob_pos = calibration_model.decision_function(x_test)
        prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    fraction_of_positives, mean_predicted_value = \
        calibration_curve(y_test, prob_pos, n_bins=10)

    sns.regplot(mean_predicted_value, fraction_of_positives, ci=95, label="%s" % (name,), ax=ax)

    ax.set_xlabel("Mean Predict Probability", fontsize=20)
    ax.set_ylabel("True Probability", fontsize=20)
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc="lower right", fontsize=15)
    ax.set_title('Calibration Curve', fontsize=20)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    ax.figure.savefig(os.path.join(save_path, 'calibration_curve.png'), dpi=300)
    plt.show()
