import numpy as np

class K_Nearest_Neighbour:
    """
    A class to implement the K-Nearest Neighbour algorithm for regression.
    """

    def __init__(self, k: int = 3):
        """
        Initializes the KNN instance with a specified number of neighbors.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit the training data.
        """
        self.X_train = X_train
        self.y_train = y_train

    @staticmethod
    def euclidean_distance(u: np.ndarray, v: np.ndarray) -> float:
        """
        Computes the Euclidean distance between two points.
        """
        return np.sqrt(np.sum((u - v) ** 2))

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predicts the target values for the test data.
        """
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x: np.ndarray) -> float:
        """
        Predicts the target value for a single test instance using the mean of the k nearest neighbors.
        """
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return np.mean(k_nearest_labels)

    def score(self, X_test, y_true):
        """
        Calculates and returns the MSE and R-squared value for the test set.
        """
        y_pred = self.predict(X_test)
        mse = np.mean((y_true - y_pred) ** 2)
        
        total_variance = np.sum((y_true - np.mean(y_true)) ** 2)
        explained_variance = np.sum((y_pred - np.mean(y_true)) ** 2)
        r_squared = explained_variance / total_variance
        
        return mse, r_squared

    def tune_and_fit(self, X_train: np.ndarray, y_train: np.ndarray, k_values: list):
        """
        Tune the K parameter over a specified range of values and fit the model with the best K.
        """
        best_k = None
        best_mse = float('inf')

        for k in k_values:
            self.k = k
            self.fit(X_train, y_train)
            mse, _ = self.score(X_train, y_train)
            print(f"Testing K={k}: MSE={mse}, R-squared={_}")  # Log progress

            if mse < best_mse:
                best_mse = mse
                best_k = k

        self.k = best_k
        self.fit(X_train, y_train)

        return best_k, best_mse
