import numpy as np

class LassoRegression:
    def __init__(self, learning_rate=0.01, no_of_iterations=1000, lambda_parameter=0.1):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.lambda_parameter = lambda_parameter
        self.weight = None

    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y).flatten()

        X = np.insert(X, 0, 1, axis=1)
        self.samples, self.features = X.shape
        self.weight = np.zeros(self.features)

        for i in range(self.no_of_iterations):
            self._update_weights(X, Y)

    def _update_weights(self, X, Y):
        Y_pred = np.dot(X, self.weight)
        dW = np.zeros(self.features)

        for j in range(self.features):
            if j == 0:
                dW[j] = -2 * np.dot(X[:, j], (Y - Y_pred)) / self.samples
            else:
                dW[j] = (-2 * np.dot(X[:, j], (Y - Y_pred)) + np.sign(self.weight[j]) * self.lambda_parameter) / self.samples

            self.weight[j] -= self.learning_rate * dW[j]

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = np.insert(X, 0, 1, axis=1)
        return np.dot(X, self.weight)

    def evaluate(self, X, Y):
        predictions = self.predict(X)
        mse = np.mean((Y - predictions) ** 2)
        ss_res = np.sum((Y - predictions) ** 2)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        return mse, r_squared

    def tune_and_fit(self, X_train, y_train, X_val, y_val, lambda_params):
        best_lambda = None
        best_mse = float('inf')
        best_r_squared = -float('inf')  # Assuming you want to track the best R-squared

        for lambda_param in lambda_params:
            # Set the current lambda parameter
            self.lambda_parameter = lambda_param

            # Fit the model on the training data
            self.fit(X_train, y_train)

            # Evaluate the model on the validation data
            predictions = self.predict(X_val)
            mse = np.mean((y_val - predictions) ** 2)
            
            # Calculate R-squared
            ss_res = np.sum((y_val - predictions) ** 2)
            ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # Print the current lambda, MSE, and R-squared
            print(f"Lambda: {lambda_param}, MSE: {mse}, R-squared: {r_squared}")

            # Check if the current mse is better than the best mse found so far
            if mse < best_mse or (mse == best_mse and r_squared > best_r_squared):
                best_mse = mse
                best_lambda = lambda_param
                best_r_squared = r_squared

        # After finding the best lambda, re-fit the model with it
        self.lambda_parameter = best_lambda
        self.fit(X_train, y_train)

        # Return the best lambda, mse, and R-squared
        return best_lambda, best_mse, best_r_squared