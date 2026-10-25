
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: CREDIT CARD MODEL                                                                          -- #
# -- script: dqr.py : python script with DQR functionality                                               -- #
# -- author: anasofiabrizuela / Antonio-IF / diegotita4 / luisrc44 / Oscar148                            -- #
# -- license: MIT License                                                                                -- #
# -- repository: https://github.com/diegotita4/credit-approval-model                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# IMPORT NECESSARY LIBRARIES
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------

# 
class DataQualityReview:
    """
    Class to perform Data Quality Review (DQR) including preprocessing, handling outliers,
    handling missing values, encoding categorical variables, and saving the cleaned data.

    Attributes:
        data_path (str): Path to the dataset.
        target_column (str): Name of the target column in the dataset.
        selected_columns (list): Columns to be used in the analysis.
        selected_columns_to_drop (list): Columns to be excluded from the analysis.
        want_impute (str): Specifies whether missing values should be imputed.
        fill_method (str): Method used for imputing missing values ('median', 'mean', 'mode').

    Methods:
        validate_data(data): Ensures that the data is not empty.
        preprocess_data(): Drops unwanted columns and prepares data for further processing.
        handle_outliers(): Identifies and handles outliers in numerical columns using IQR.
        handle_missing_values(): Handles missing values according to specified imputation method.
        encode_categorical(): Encodes categorical variables into numerical values and stores mappings.
        save_cleaned_data(output_path): Saves the cleaned and processed data to a specified path.
        perform_DQR(): Executes all steps of the data quality review process.
    """

    # ------------------------------

    def __init__(self, data_path, target_column, selected_columns, selected_columns_to_drop, want_impute, fill_method):
        """
        Initializes DataQualityReview with data, configuration for data processing, and prepares the initial dataset.
        """

        self.data = pd.read_excel(data_path)[selected_columns]
        self.validate_data(self.data)
        self.target_column = target_column
        self.selected_columns = selected_columns
        self.selected_columns_to_drop = selected_columns_to_drop
        self.want_impute = want_impute
        self.fill_method = fill_method

    # ------------------------------

    def validate_data(self, data):
        """
        Validates that the provided DataFrame is not empty, raises a ValueError if it is.
        """

        if data.empty:
            raise ValueError('\nDataFrame must be not empty.')

    # ------------------------------

    def preprocess_data(self):
        """
        Removes specified columns from the data and prints a summary of the action.
        """

        print(f"\n{'='*50}\nPreporcess data...\n{'='*50}")

        try:
            if self.selected_columns_to_drop:
                self.data.drop(columns=self.selected_columns_to_drop, errors='ignore', inplace=True)
            else:
                print(f'\nNo columns to drop.')

        except KeyError as e:
            print(f'\nError dropping columns: {e}')

    # ------------------------------

    def handle_outliers(self):
        """
        Handles outliers by clipping values outside 1.5 times the IQR from the Q1 and Q3 of each numerical column.
        """

        print(f"\n{'='*50}\nHandle outliers...\n{'='*50}")

        try:
            for col in self.data.select_dtypes(include=['number']).columns:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                self.data[col] = self.data[col].clip(lower=lower_bound, upper=upper_bound)

        except Exception as e:
            print(f'\nError handling outliers: {e}')

    # ------------------------------

    def handle_missing_values(self):
        """
        Handles missing values either by dropping rows where all columns are NaN or imputing based on a specified method.
        """

        print(f"\n{'='*50}\nHandle missing values...\n{'='*50}")

        try:
            self.data.dropna(how='all', subset=[col for col in self.data.columns if col != self.target_column], inplace=True)

            if self.want_impute.lower() == 'yes':
                for column in self.data.columns:
                    if column != self.target_column:
                        if self.data[column].dtype in ['float64', 'int64'] and self.fill_method == 'median':
                            self.data[column].fillna(self.data[column].median(), inplace=True)

                        elif self.data[column].dtype in ['float64', 'int64'] and self.fill_method == 'mean':
                            self.data[column].fillna(self.data[column].median(), inplace=True)

                        elif self.data[column].dtype == 'object':
                            self.data[column].fillna(self.data[column].mode()[0], inplace=True)

        except Exception as e:
            print(f'\nError handling missing values: {e}')
            
    # ------------------------------

    def encode_categorical(self):
        """
        Encodes categorical variables into numerical values using LabelEncoder and stores mappings.
        """

        print(f"\n{'='*50}\nEncoding categorical...\n{'='*50}")

        try:
            self.label_encoders = {}

            for column in self.data.select_dtypes(include=['object']).columns:
                if column != self.target_column:
                    encoder = LabelEncoder()
                    self.data[column] = encoder.fit_transform(self.data[column].astype(str))
                    self.label_encoders[column] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                    print(f'\n{self.label_encoders}')

        except Exception as e:
            print(f'\nError encoding categorical variables: {e}')

    # ------------------------------

    def save_cleaned_data(self, output_path='data/clean_data.xlsx'):
        """
        Saves the cleaned data to a specified file path after creating necessary directories.
        """

        print(f"\n{'='*50}\nSaving cleaned data...\n{'='*50}")

        try:
            model_directory = os.path.dirname(output_path)
            
            if not os.path.exists(model_directory):
                os.makedirs(model_directory)
                print(f"\n¡Created directory: '{model_directory}'!\n\n{'-'*50}")

            self.data.to_excel(output_path, index=False)
            print(f"\n¡Cleaned data saved: '{output_path}'!")

        except Exception as e:
            print(f'\nError saving cleaned data: {e}')

    # ------------------------------

    def perform_DQR(self):
        """
        Executes the entire data quality review process in sequence.
        """

        self.preprocess_data()
        self.handle_outliers()
        self.handle_missing_values()
        self.encode_categorical()
        self.save_cleaned_data()

        return self.data
