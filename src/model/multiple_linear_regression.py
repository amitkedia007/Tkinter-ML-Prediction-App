import numpy as np
import pandas as pd

class MultipleLinearRegression:
    def __init__(self):
        self.coef_ = None  # Coefficients of the model
        self.intercept_ = None  # Intercept of the model

    def fit(self, X_train, y_train):
        """
        Fits the multiple linear regression model to the training data.
        """
        # Convert pandas DataFrame or Series to numpy arrays
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_train = y_train.values.flatten()

        # Add an intercept column to the input features
        X_train = np.insert(X_train, 0, 1, axis=1)

        # Calculate coefficients using the Normal Equation
        betas = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

        # Assign intercept and coefficients
        self.intercept_ = betas[0]
        self.coef_ = betas[1:]

    def predict(self, X_test):
        """
        Predicts the target values for the given input features using the trained model.
        """
        # Convert pandas DataFrame to numpy array
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values

        # Add an intercept column to the input features
        X_test = np.insert(X_test, 0, 1, axis=1)

        # Compute predictions
        y_pred = X_test @ np.hstack(([self.intercept_], self.coef_))
        return y_pred

    def score(self, X_test, y_true):
        """
        Computes the Mean Squared Error (MSE) and R-squared value of the model
        against the provided test data and true output values.
        """
        # Convert pandas Series or DataFrame to numpy array
        if isinstance(y_true, (pd.Series, pd.DataFrame)):
            y_true = y_true.values.flatten()

        # Get model predictions
        y_pred = self.predict(X_test)

        # Calculate MSE
        mse = np.mean((y_true - y_pred) ** 2)

        # Calculate R-squared
        total_variance = np.sum((y_true - np.mean(y_true)) ** 2)
        residual_variance = np.sum((y_true - y_pred) ** 2)
        r_squared = 1 - (residual_variance / total_variance)

        return mse, r_squared
