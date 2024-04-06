import numpy as np
import pandas as pd

class RidgeRegression:
    
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X_train, y_train):
        # Ensure X_train and y_train are numpy arrays
        X_train, y_train = self._ensure_numpy(X_train, y_train)
        
        X_train_with_intercept = np.insert(X_train, 0, 1, axis=1)
        I_matrix = np.identity(X_train_with_intercept.shape[1])
        I_matrix[0, 0] = 0  # Do not regularize the intercept term
        equation = np.linalg.inv(X_train_with_intercept.T @ X_train_with_intercept + self.alpha * I_matrix) @ X_train_with_intercept.T @ y_train
        
        self.intercept_ = equation[0]
        self.coef_ = equation[1:]

    def predict(self, X_test):
        # Ensure X_test is a numpy array
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy()
        
        return np.dot(X_test, self.coef_) + self.intercept_

    def score(self, X_test, y_true):
        # Ensure X_test and y_true are numpy arrays
        X_test, y_true = self._ensure_numpy(X_test, y_true)

        y_pred = self.predict(X_test)
        mse = np.mean((y_true - y_pred) ** 2)
        total_variance = np.sum((y_true - np.mean(y_true)) ** 2)
        explained_variance = np.sum((y_pred - np.mean(y_true)) ** 2)
        r_squared = explained_variance / total_variance

        return mse, r_squared

    def tune_and_fit(self, X_train, y_train, alphas):
        best_alpha = None
        best_score = float('inf')
        best_r_squared = None

        X_train, y_train = self._ensure_numpy(X_train, y_train)
        print("Tuning Ridge Regression with the following alpha values:", alphas)
        for alpha in alphas:
            self.alpha = alpha
            self.fit(X_train, y_train)
            mse, r_squared = self.score(X_train, y_train)
            if mse < best_score:
                best_score = mse
                best_alpha = alpha
                best_r_squared = r_squared
            print(f"Alpha: {alpha:10}, MSE: {mse:.4f}, R-squared: {r_squared:.4f}")  
        self.alpha = best_alpha
        self.fit(X_train, y_train)
        return best_alpha, best_score, best_r_squared

    def _ensure_numpy(self, X, y):
        """Ensure X and y are numpy arrays."""
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.to_numpy()
        return X, y
