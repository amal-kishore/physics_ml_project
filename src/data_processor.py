import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        # Load data from Excel file
        data = pd.read_excel(self.data_path)

        # Ensure all columns are numeric or categorical
        for col in data.columns:
            if data[col].dtype == 'object':
                try:
                    # Convert object columns to categorical if possible
                    data[col] = data[col].astype('category')
                except:
                    # Handle text columns that can't be converted to categories
                    print(f"Removing non-numeric column: {col}")
                    data = data.drop(columns=[col])
        
        return data

    def prepare_data(self):
        # Load and preprocess data
        data = self.load_data()
        
        # Separate features (X) and target (y)
        X = data.drop("gap_hse", axis=1)  # Adjust column name if needed
        y = data["gap_hse"]               # Adjust column name if needed
        
        # Split data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        return X_train, X_val, X_test, y_train, y_val, y_test

