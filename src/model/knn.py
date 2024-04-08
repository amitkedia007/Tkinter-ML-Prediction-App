import numpy as np

class K_Nearest_Neighbour:
    def __init__(self, k=3):
        self.k = k

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = self.euclidean_distance(self.X_train, x)
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            prediction = np.mean(k_nearest_labels)
            predictions.append(prediction)
        return np.array(predictions)

    def score(self, X_test, y_true):
        """
        Calculates and returns the MSE and R-squared value for the test set.
        """
        y_pred = self.predict(X_test)
        mse = np.mean((y_true - y_pred) ** 2)
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return mse, r_squared


    def cross_validate(self, X, y, k_values, n_splits=5):
        fold_scores_mse = {k: [] for k in k_values}
        fold_scores_r2 = {k: [] for k in k_values}

        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)

        fold_sizes = np.full(n_splits, len(indices) // n_splits, dtype=int)
        fold_sizes[:len(indices) % n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            for k in k_values:
                self.k = k
                self.fit(X_train, y_train)
                mse, r_squared = self.score(X_test, y_test)
                fold_scores_mse[k].append(mse)
                fold_scores_r2[k].append(r_squared)

            current = stop

        best_k, best_score_mse, best_score_r2 = None, float('inf'), None

        for k in k_values:
            avg_mse = np.mean(fold_scores_mse[k])
            avg_r2 = np.mean(fold_scores_r2[k])
            print(f"K: {k}, Avg MSE: {avg_mse}, Avg R-squared: {avg_r2}")

            if avg_mse < best_score_mse:
                best_k = k
                best_score_mse = avg_mse
                best_score_r2 = avg_r2

        print(f"Best K: {best_k} with Avg MSE: {best_score_mse}, Avg R-squared: {best_score_r2}")
        return best_k, best_score_mse, best_score_r2
