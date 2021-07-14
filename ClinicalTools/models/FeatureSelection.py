import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')


def lasso_sele(x_train, y_train):
    lassocv = LassoCV()
    lassocv.fit(x_train, y_train)
    alpha = lassocv.alpha_
    print('利用Lasso交叉检验计算得出的最优alpha：' + str(alpha))
    n = np.sum(lassocv.coef_ != 0)  # 计算系数不为0的个数
    print('Lasso回归后系数不为0的个数: ' + str(n))
    lasso_select_vars = list(x_train.columns[lassocv.coef_ != 0])
    return lasso_select_vars


def rfe_sele(x_train, y_train, model='defalut', savefile='rfe_pic.png'):
    if model == 'defalut':
        model = RandomForestClassifier(criterion='gini', n_estimators=10,
                                       class_weight='balanced', random_state=42, n_jobs=-1)

    rfe = RFECV(model, cv=StratifiedKFold(5),
                scoring='roc_auc', n_jobs=-1, step=1)
    rfe.fit(x_train, y_train)
    print('Have {} variables have been selected.'.format(len(x_train.columns[rfe.support_.tolist()])))

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_, c='black')
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)
    ax.set_xlabel("Number of features selected", fontsize=25, fontdict={'weight': 'bold'})
    ax.set_ylabel("Cross validation score (AUC)", fontsize=25, fontdict={'weight': 'bold'})
    ax.set_title('Number of Features vs AUC', fontsize=25, fontdict={'weight': 'bold'})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('{}.png'.format(savefile), dpi=300)
    print('The picture of REF result have been save as "{}"'.format(savefile))
    plt.show()

    return x_train.columns[rfe.support_.tolist()]
