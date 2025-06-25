# features/force_transformer.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ForceFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n=10, sigma=5.0):
        self.n = n
        self.sigma = sigma

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        close = X['Close'].values
        volume = X['Volume'].values
        force_list = []


        for i in range(len(X)):
            if i < self.n:
                force_list.append(0.0)
                continue

            p = close[i]
            past_prices = close[i - self.n:i]
            past_volumes = volume[i - self.n:i]
            V = past_volumes / (np.sum(past_volumes) + 1e-8)

            F = 0.0
            for j in range(self.n):
                delta = p - past_prices[j]
                kernel = np.exp(- (delta ** 2) / (2 * self.sigma ** 2))
                F_j = -V[j] * (delta / self.sigma ** 2) * kernel

                F += F_j

            force_list.append(F)

        return pd.DataFrame({'force': force_list})