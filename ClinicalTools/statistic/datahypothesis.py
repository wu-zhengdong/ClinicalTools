# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 09:26:48 2021

@author: FengY Z
"""
import pandas as pd
import numpy as np
import warnings
from scipy.stats import ks_2samp
from scipy.stats import ttest_ind
from scipy.stats import levene

warnings.filterwarnings('ignore')


def base_sta(train, test, classes=5):
    compared_vars = [n for n in train.columns if len(train[n].value_counts()) <= classes]
    cont_var = [n for n in train.columns if len(train[n].value_counts()) > classes]

    p_value_dict = {}
    all_df = pd.DataFrame()
    for var in compared_vars:
        train_fenbu = train[var].value_counts().reset_index()
        train_fenbu.rename(columns={var: 'train'}, inplace=True)
        train_fenbu_zhanbi = train[var].value_counts(normalize=True).reset_index()
        train_fenbu_zhanbi.rename(columns={var: 'train_rate'}, inplace=True)
        test_fenbu = test[var].value_counts().reset_index()
        test_fenbu.rename(columns={var: 'test'}, inplace=True)
        test_fenbu_zhanbi = test[var].value_counts(normalize=True).reset_index()
        test_fenbu_zhanbi.rename(columns={var: 'test_rate'}, inplace=True)
        for df in [train_fenbu_zhanbi, test_fenbu, test_fenbu_zhanbi]:
            train_fenbu = pd.merge(train_fenbu, df, on='index', how='outer')
        train_fenbu.rename(columns={'index': 'Subtype'}, inplace=True)
        train_fenbu['Characteristics'] = [var] * len(train_fenbu)
        all_df = all_df.append(train_fenbu)
        ks_test, p_value = ks_2samp(train[var], test[var])
        p_value_dict[var] = [p_value]

    all_df['train_rate'] = all_df['train_rate'].apply(lambda x: '(' + str(round(x * 100, 2)) + '%' + ')')
    all_df['test_rate'] = all_df['test_rate'].apply(lambda x: '(' + str(round(x * 100, 2)) + '%' + ')')
    all_df['train'] = all_df['train'].apply(lambda x: str(x))
    all_df['test'] = all_df['test'].apply(lambda x: str(x))
    all_df['train'] = all_df['train'] + all_df['train_rate']
    all_df['test'] = all_df['test'] + all_df['test_rate']
    all_df.drop(['train_rate', 'test_rate'], axis=1, inplace=True)
    p_value_df = pd.DataFrame(p_value_dict).T.reset_index()
    p_value_df.rename(columns={'index': 'Characteristics', 0: 'p_values'}, inplace=True)
    all_df = all_df.merge(p_value_df, on='Characteristics', how='left')
    all_df = all_df[['Characteristics', 'Subtype', 'train', 'test', 'p_values']]
    new_all_df = pd.DataFrame()
    for df in all_df.groupby('Characteristics'):
        tmp = df[1].sort_values(by='Subtype')
        new_all_df = new_all_df.append(tmp)
    result_dict = {}

    for var in cont_var:
        train_mean = train[var].mean()
        test_mean = test[var].mean()

        train_std = train[var].std()
        test_std = test[var].std()

        _, levene_p = levene(train[var], test[var])
        #         print(levene_p)
        if levene_p < 0.05:
            _, p_value = ttest_ind(train[var], test[var], equal_var=False)
        else:
            _, p_value = ttest_ind(train[var], test[var])
        result_dict[var] = ["{}({})".format(round(train_mean, 2), round(train_std, 2)),
                            "{}({})".format(round(test_mean, 2), round(test_std, 2)), round(p_value, 3)]

    con_df = pd.DataFrame(result_dict).T.reset_index()
    con_df.rename(columns={'index': 'Characteristics', 0: 'train', 1: 'test', 2: 'p_values'}, inplace=True)
    con_df['Subtype'] = [np.nan] * len(con_df)
    con_df = con_df[['Characteristics', 'Subtype', 'train', 'test', 'p_values']]
    sta_df = pd.concat([new_all_df, con_df], axis=0)
    return sta_df