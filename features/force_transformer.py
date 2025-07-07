# features/force_transformer.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ForceFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n=10, sigma=1.0):
        self.n = n
        self.sigma = sigma

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        force_list, potential_list = [], []

        for i in range(len(df)):
            if i < self.n:
                force_list.append(0)
                potential_list.append(0)
                continue

            p = df['Close'].iloc[i]
            P_i = df['Close'].iloc[i - self.n:i].values
            V_i = df['Volume'].iloc[i - self.n:i].values
            V_i = V_i / np.sum(V_i)  # 归一化

            # U(P) and F(P)
            kernel = np.exp(-((p - P_i) ** 2) / (2 * self.sigma ** 2))
            U = np.sum(V_i * kernel)
            F = -np.sum(V_i * (p - P_i) / self.sigma ** 2 * kernel)

            force_list.append(F)
            potential_list.append(U)

        df['force'] = force_list
        df['potential'] = potential_list

        return df[['force', 'potential']]