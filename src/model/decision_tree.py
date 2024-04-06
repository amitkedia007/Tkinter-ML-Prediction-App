import numpy as np

class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth if max_depth is not None else float('inf')
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    class Node:
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        # Handle None for max_depth by using a large value for comparison
        max_depth = float('inf') if self.max_depth is None else self.max_depth
        
        if num_samples < self.min_samples_split or depth >= max_depth:
            return self.Node(value=self._calculate_leaf_value(y))
        
        best_split = self._get_best_split(X, y, num_samples, num_features)
        if best_split is None:
            return self.Node(value=self._calculate_leaf_value(y))
        
        # It looks like your method for accessing split indices might also need adjustment
        # Assuming best_split["dataset_left"] and best_split["dataset_right"] are boolean masks or indices arrays
        left_indices, right_indices = best_split["dataset_left"], best_split["dataset_right"]
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return self.Node(feature_index=best_split["feature_index"], threshold=best_split["threshold"],
                        left=left_subtree, right=right_subtree)


    def _get_best_split(self, X, y, num_samples, num_features):
        best_split = {"gain": -float("inf")}
        for feature_index in range(num_features):
            feature_values = X[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                dataset_left, labels_left, dataset_right, labels_right = self._split(X, y, feature_index, threshold)
                if len(labels_left) > 0 and len(labels_right) > 0:
                    current_gain = self._information_gain(y, labels_left, labels_right)
                    if current_gain > best_split["gain"]:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = np.where(X[:, feature_index] <= threshold)[0]
                        best_split["labels_left"] = labels_left
                        best_split["dataset_right"] = np.where(X[:, feature_index] > threshold)[0]
                        best_split["labels_right"] = labels_right
                        best_split["gain"] = current_gain
        return best_split if best_split["gain"] > 0 else None


    def _split(self, X, y, feature_index, threshold):
        left_indices = np.where(X[:, feature_index] <= threshold)[0]
        right_indices = np.where(X[:, feature_index] > threshold)[0]
        return X[left_indices], y[left_indices], X[right_indices], y[right_indices]

    def _information_gain(self, parent, left_child, right_child):
        weight_l = len(left_child) / len(parent)
        weight_r = len(right_child) / len(parent)
        gain = self._mean_squared_error(parent) - (weight_l * self._mean_squared_error(left_child) + weight_r * self._mean_squared_error(right_child))
        return gain

    def _mean_squared_error(self, y):
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

    def _calculate_leaf_value(self, y):
        return np.mean(y)

    def predict(self, X):
        return np.array([self._make_prediction(x, self.tree) for x in X])

    def _make_prediction(self, x, tree):
        if tree.value is not None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self._make_prediction(x, tree.left)
        else:
            return self._make_prediction(x, tree.right)

    def tune_and_fit(self, X_train, y_train, X_val, y_val, max_depth_values):
        best_max_depth = None
        best_mse = float('inf')
        best_r_squared = -float('inf')  # Initialize with a very low R-squared
        
        for depth in max_depth_values:
            self.max_depth = depth
            self.fit(X_train, y_train)
            predictions = self.predict(X_val)
            mse = np.mean((y_val - predictions) ** 2)
            # Calculate R-squared
            ss_res = np.sum((y_val - predictions) ** 2)
            ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Log for debugging
            print(f"Testing max_depth={depth}: MSE={mse}, R-squared={r_squared}")
            
            if mse < best_mse or (mse == best_mse and r_squared > best_r_squared):
                best_mse = mse
                best_max_depth = depth
                best_r_squared = r_squared
        
        # Re-fit the model with the best parameters found
        self.max_depth = best_max_depth
        self.fit(X_train, y_train)
        
        return best_max_depth, best_mse, best_r_squared

    
    def score(self, X_test, y_true):
        y_pred = self.predict(X_test)
        mse = np.mean((y_true - y_pred) ** 2)
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return mse, r_squared

