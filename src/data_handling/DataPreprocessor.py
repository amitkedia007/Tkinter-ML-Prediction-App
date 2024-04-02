import pandas as pd

class DataPreprocessor:
    def __init__(self, dataframe):
        """
        Initialize the DataPreprocessor with a pandas DataFrame.
        :param dataframe: pandas DataFrame to be preprocessed.
        """
        self.dataframe = dataframe

    def fill_mean(self, column):
        """
        Fill missing values with the mean for the specified column.
        :param column: column name to fill with mean.
        """
        if column in self.dataframe.columns and self.dataframe[column].dtype in ['int64', 'float64']:
            mean_value = self.dataframe[column].mean()
            self.dataframe[column].fillna(mean_value, inplace=True)

    def fill_unknown(self, column):
        """
        Fill missing values with 'Unknown' for the specified column.
        :param column: column name to fill with 'Unknown'.
        """
        if column in self.dataframe.columns:
            self.dataframe[column].fillna('Unknown', inplace=True)

    def drop_columns(self, column):
        """
        Drop the specified column from the dataframe.
        :param column: column name to drop.
        """
        if column in self.dataframe.columns:
            self.dataframe.drop(columns=[column], axis=1, inplace=True)

    def scale_data(self):
        """
        Scale all numeric columns using min-max scaling.
        """
        numeric_columns = self.dataframe.select_dtypes(include=['int64', 'float64']).columns
        for column in numeric_columns:
            min_value = self.dataframe[column].min()
            max_value = self.dataframe[column].max()
            range_value = max_value - min_value
            if range_value > 0:
                self.dataframe[column] = (self.dataframe[column] - min_value) / range_value
            else:
                self.dataframe[column] = 0  # Handle the case where min and max are the same

    def convert_categorical(self, column):
        """
        Convert a categorical column to numeric using one-hot encoding.
        :param column: column name to encode.
        """
        if column in self.dataframe.columns and self.dataframe[column].dtype == 'object':
            one_hot = pd.get_dummies(self.dataframe[column], prefix=column)
            self.dataframe = pd.concat([self.dataframe, one_hot], axis=1)
            self.dataframe.drop(columns=[column], axis=1, inplace=True)
