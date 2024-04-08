import numpy as np

class RidgeRegression:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.weights = None

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)  # Add intercept
        I = np.eye(X.shape[1])
        I[0, 0] = 0  # Do not regularize the intercept
        self.weights = np.linalg.inv(X.T @ X + self.alpha * I) @ X.T @ y

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)  # Add intercept
        return X @ self.weights

    def score(self, X, y):
        predictions = self.predict(X)
        mse = np.mean((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - predictions) ** 2)
        r_squared = 1 - ss_res / ss_tot
        return mse, r_squared

    def k_fold_cross_validation(self, X, y, k=5, alphas=[0.1]):
        fold_size = len(X) // k
        best_alpha = None
        best_score = float('inf')
        best_r_squared = None

        for alpha in alphas:
            mse_scores = []
            r2_scores = []
            for fold in range(k):
                start, end = fold * fold_size, (fold + 1) * fold_size
                X_val, y_val = X[start:end], y[start:end]
                X_train = np.concatenate([X[:start], X[end:]])
                y_train = np.concatenate([y[:start], y[end:]])
                
                self.alpha = alpha
                self.fit(X_train, y_train)
                mse, r_squared = self.score(X_val, y_val)

                mse_scores.append(mse)
                r2_scores.append(r_squared)

            avg_mse = np.mean(mse_scores)
            avg_r_squared = np.mean(r2_scores)

            if avg_mse < best_score:
                best_score = avg_mse
                best_alpha = alpha
                best_r_squared = avg_r_squared
            
            print(f"Alpha: {alpha}, Avg MSE: {avg_mse:.4f}, Avg R-squared: {avg_r_squared:.4f}")

        self.alpha = best_alpha
        self.fit(X, y)  # Refit using the best alpha
        return best_alpha, best_score, best_r_squared
