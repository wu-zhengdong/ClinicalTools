import pandas as pd
import numpy as np


class StandardScaler:
    def __init__(self, threshold=5):
        self.threshold = threshold
        self.standard_df = pd.DataFrame([])

        self.continue_columns = None

    def fit(self, x_train):
        self.continue_columns = [n for n in x_train.columns if len(x_train[n].value_counts()) > self.threshold]
        # z = (x - u) / s
        mean_df = pd.DataFrame(np.mean(x_train[self.continue_columns])).rename(columns={0: 'mean'}).T
        std_df = pd.DataFrame(np.std(x_train[self.continue_columns])).rename(columns={0: 'std'}).T
        self.standard_df = pd.concat([mean_df, std_df], axis=0)

    def transform(self, x):
        assert self.standard_df is not None, 'Pls fit before transform.'
        for n in self.continue_columns:
            u = self.standard_df.loc['mean', n]
            s = self.standard_df.loc['std', n]
            x[n] = (x[n] - u) / s

        return x

    def inverse_transform(self, x):
        assert self.standard_df is not None, 'Pls fit before transform.'
        for n in self.continue_columns:
            u = self.standard_df.loc['mean', n]
            s = self.standard_df.loc['std', n]
            x[n] = x[n] * s + u
        return x
