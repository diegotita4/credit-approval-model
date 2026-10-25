
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: CREDIT CARD MODEL                                                                          -- #
# -- script: main.py : python script with MAIN functionality                                             -- #
# -- author: anasofiabrizuela / Antonio-IF / diegotita4 / luisrc44 / Oscar148                            -- #
# -- license: MIT License                                                                                -- #
# -- repository: https://github.com/diegotita4/credit-approval-model                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# IMPORT NECESSARY MODULES FROM OTHER SCRIPTS
from dqr import DataQualityReview
from model import CreditCardModel

# --------------------------------------------------

# MAIN EXECUTION BLOCK
if __name__ == '__main__':
    """
    This script is the entry point for running the Credit Card Approval Model. It orchestrates the flow of data from raw to processed,
    executes data quality reviews, performs exploratory data analysis, prepares the data, builds, trains, and evaluates the model,
    and finally saves the trained model to a file.

    Args:
        data_path (str): Path to the raw data file.
        target_column (str): The name of the target column in the dataset.
        selected_columns (list): List of column names to be included in the analysis and model training.
        selected_columns_to_drop (list): List of column names to be excluded from the model training.
        want_impute (str): Indicator whether to perform imputation ('yes' or 'no').
        fill_method (str): Method to use for imputing missing values ('mode', 'median', 'mean').
        clean_data_path (str): Path where the cleaned data will be saved.
        model_type (str): Type of model to train ('logistic_regression', 'random_forest', 'xgboost', 'neural_network').

    Returns:
        None: This script does not return any value but will print outputs related to the process stages and save the trained model to disk.
    """

    data_path = 'data/data.xlsx'

    target_column = 'ESTATUS'
    selected_columns = ['Ingreso_Estimado', 'Egresos', 'Capacidad_Pago', 'Endeudamiento', 'ESTATUS']
    selected_columns_to_drop = []

    want_impute = 'no'   # OPTIONS: 'yes' OR 'no'
    fill_method = 'median'   # OPTIONS: 'mode', 'median', 'mean'

    data_quality_review = DataQualityReview(data_path, target_column, selected_columns, selected_columns_to_drop, want_impute, fill_method)
    data_quality_review.perform_DQR()
    clean_data_path = 'data/clean_data.xlsx'

    use_clean_data = 'no'
    model_type = 'logistic_regression'   # OPTIONS: 'logistic_regression', 'random_forest', 'xgboost', 'neural_network'

    if use_clean_data == 'yes':
        credit_card_model = CreditCardModel(clean_data_path, target_column, selected_columns, selected_columns_to_drop, model_type)
        credit_card_model.eda(clean_data_path)

    else:
        credit_card_model = CreditCardModel(data_path, target_column, selected_columns, selected_columns_to_drop, model_type)
        credit_card_model.eda(data_path)

    credit_card_model.prepare_data()
    credit_card_model.build_model()

    if model_type == 'neural_network':
        history = credit_card_model.train_model(epochs=30, batch_size=32)

    else:
        credit_card_model.train_model()

    credit_card_model.evaluate_model()
    credit_card_model.save_model()
