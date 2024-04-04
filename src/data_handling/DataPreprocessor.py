import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        

    def fill_missing_values(self):
        for column in self.dataframe.columns:
            if self.dataframe[column].dtype in ['int64', 'float64']:
                self.dataframe.loc[:, column].fillna(self.dataframe[column].mean(), inplace=True)
            else:
                self.dataframe[column].fillna(self.dataframe[column].mode()[0], inplace=True)

    def drop_highly_correlated_columns(self, threshold=0.95):
        corr_matrix = self.dataframe.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        self.dataframe.drop(columns=to_drop, inplace=True)

    def encode_categorical_variables(self):
        categorical_cols = self.dataframe.select_dtypes(include=['object', 'category']).columns
        self.dataframe = pd.get_dummies(self.dataframe, columns=categorical_cols, drop_first=True)

    def apply_min_max_scaling(self):
        numeric_cols = self.dataframe.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            min_val = self.dataframe[col].min()
            max_val = self.dataframe[col].max()
            self.dataframe[col] = (self.dataframe[col] - min_val) / (max_val - min_val)

    def preprocess_data(self, drop_correlated=True, scale_data=True):
        self.fill_missing_values()
        if drop_correlated:
            self.drop_highly_correlated_columns()
        self.encode_categorical_variables()
        if scale_data:
            self.apply_min_max_scaling()
        return self.dataframe

    def preprocess_target(self, target_dataframe, fill_strategy="mean"):
        """Specific preprocessing for the target variable."""
        if fill_strategy == "mean" and target_dataframe.dtype in ['int64', 'float64']:
            return target_dataframe.fillna(target_dataframe.mean(), inplace=False)
        elif fill_strategy == "mode":
            return target_dataframe.fillna(target_dataframe.mode()[0], inplace=False)
        # Add more conditions as needed for different types of targets
        return target_dataframe
