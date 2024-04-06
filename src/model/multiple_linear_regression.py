import numpy as np
import pandas as pd

class MultipleLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X_train, y_train):
        # Check if X_train is a DataFrame and convert to NumPy array
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        # Check if y_train is a Series or DataFrame and convert to NumPy array
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_train = y_train.values.flatten()

        # Insert a column of ones to X_train for the intercept
        X_train = np.insert(X_train, 0, 1, axis=1)

        # Calculate betas
        betas = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

        # The first beta is the intercept
        self.intercept_ = betas[0]
        # The remaining betas are the coefficients
        self.coef_ = betas[1:]

    def predict(self, X_test):
        # Check if X_test is a DataFrame and convert to NumPy array
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values

        # Insert a column of ones to X_test for the intercept
        X_test = np.insert(X_test, 0, 1, axis=1)

        # Calculate predictions
        y_pred = X_test @ np.hstack(([self.intercept_], self.coef_))
        return y_pred

    def score(self, X_test, y_true):
        # Check if y_true is a Series or DataFrame and convert to NumPy array
        if isinstance(y_true, (pd.Series, pd.DataFrame)):
            y_true = y_true.values.flatten()

        y_pred = self.predict(X_test)
        mse = np.mean((y_true - y_pred) ** 2)
        r_squared = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        return mse, r_squared
