import numpy as np
import pandas as pd

class RidgeRegression:
    
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X_train, y_train):
        # Convert pandas DataFrame to numpy array if necessary
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_train = y_train.to_numpy()
        
        X_train_with_intercept = np.insert(X_train, 0, 1, axis=1)
        I_matrix = np.identity(X_train_with_intercept.shape[1])
        I_matrix[0, 0] = 0  # Do not regularize the intercept term
        equation = np.linalg.inv(X_train_with_intercept.T @ X_train_with_intercept + self.alpha * I_matrix) @ X_train_with_intercept.T @ y_train
        
        self.intercept_ = equation[0]
        self.coef_ = equation[1:]


    def predict(self, X_test):
        # Convert pandas DataFrame to numpy array if necessary
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy()
        
        # Predict target values
        return np.dot(X_test, self.coef_) + self.intercept_

    def score(self, X_test, y_true):
        y_pred = self.predict(X_test)
        mse = np.mean((y_true - y_pred) ** 2)  # Ensure mse is scalar
        r_squared = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)  # Ensure r_squared is scalar

        return mse, r_squared

    
    def tune_and_fit(self, X_train, y_train, alphas):
        best_alpha = None
        best_score = float('inf')
        best_r_squared = None  # Track the best R-squared value

        for alpha in alphas:
            self.alpha = alpha
            self.fit(X_train, y_train)
            mse, r_squared = self.score(X_train, y_train)  # Ensure score returns both MSE and R-squared
            print(f"Testing alpha={alpha}: MSE={mse}, R-squared={r_squared}")  # Log progress

            mse = mse.item() if isinstance(mse, pd.Series) else mse

            if mse < best_score:
                best_score = mse
                best_alpha = alpha
                best_r_squared = r_squared  # Update the best R-squared

        self.alpha = best_alpha
        self.fit(X_train, y_train)
        return best_alpha, best_score, best_r_squared  # Return best R-squared along with alpha and MSE

