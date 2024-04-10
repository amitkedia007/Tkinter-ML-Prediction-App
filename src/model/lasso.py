import numpy as np

class LassoRegression:
    def __init__(self, learning_rate=0.01, no_of_iterations=1000, lambda_parameter=0.1):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.lambda_parameter = lambda_parameter
        self.weight = None

    def fit(self, X, Y):
        X = np.insert(X, 0, 1, axis=1)
        self.samples, self.features = X.shape
        self.weight = np.zeros(self.features)
        for _ in range(self.no_of_iterations):
            self._update_weights(X, Y)

    def _update_weights(self, X, Y):
        Y_pred = np.dot(X, self.weight)
        dW = np.zeros(self.features)
        for j in range(self.features):
            if j == 0:
                dW[j] = -2 * np.dot(X[:, j], (Y - Y_pred)) / self.samples
            else:
                dW[j] = (-2 * np.dot(X[:, j], (Y - Y_pred)) + 2 * self.lambda_parameter * np.sign(self.weight[j])) / self.samples
            self.weight[j] -= self.learning_rate * dW[j]

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return np.dot(X, self.weight)

    def evaluate(self, X, Y):
        predictions = self.predict(X)
        mse = np.mean((Y - predictions) ** 2)
        ss_res = np.sum((Y - predictions) ** 2)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)
        r_squared = 1 - (ss_res/ss_tot)
        return mse, r_squared

    def tune_and_fit(self, X, y, lambda_params, k_folds):
        best_lambda = None
        best_mse = float('inf')
        best_r_squared = -float('inf')
        tuning_details = ""
        lambdas = []
        mses = []

        for lambda_param in lambda_params:
            self.lambda_parameter = lambda_param
            mse_list = []
            r_squared_list = []

            for k in range(k_folds):
                start, end = (len(X) * k // k_folds, len(X) * (k + 1) // k_folds)
                X_val, y_val = X[start:end], y[start:end]
                X_train = np.concatenate((X[:start], X[end:]))
                y_train = np.concatenate((y[:start], y[end:]))

                self.fit(X_train, y_train)
                mse, r_squared = self.evaluate(X_val, y_val)

                mse_list.append(mse)
                r_squared_list.append(r_squared)

            avg_mse = np.mean(mse_list)
            avg_r_squared = np.mean(r_squared_list)
            tuning_details += f"Lambda: {lambda_param}, Avg MSE: {avg_mse:.4f}, Avg R-squared: {avg_r_squared:.4f}\n"
            lambdas.append(lambda_param)
            mses.append(avg_mse)

            if avg_mse < best_mse or (avg_mse == best_mse and avg_r_squared > best_r_squared):
                best_mse = avg_mse
                best_lambda = lambda_param
                best_r_squared = avg_r_squared

        # Correct return statement to match expected unpacking
        return best_lambda, best_mse, best_r_squared, tuning_details, lambdas, mses
