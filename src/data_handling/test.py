import pandas as pd
import numpy as np
from DataPreprocessor import DataPreprocessor
# Sample synthetic dataset
data = {
    'Age': [25, 32, np.nan, 45, 22],
    'Income': [50000, 60000, 52000, np.nan, 58000],
    'Gender': ['Male', 'Female', 'Female', 'Male', np.nan],
    'Occupation': ['Engineer', 'Doctor', 'Artist', 'Engineer', 'Doctor']
}

df = pd.read_csv("HousingData.csv", delimiter=',')

preprocessor = DataPreprocessor(df)

# Preprocess the data
preprocessed_df = preprocessor.preprocess_data()

# Display the preprocessed DataFrame
print(preprocessed_df.head(15))

