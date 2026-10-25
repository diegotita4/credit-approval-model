
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: CREDIT CARD MODEL                                                                          -- #
# -- script: model.py : python script with MODEL functionality                                           -- #
# -- author: anasofiabrizuela / Antonio-IF / diegotita4 / luisrc44 / Oscar148                            -- #
# -- license: MIT License                                                                                -- #
# -- repository: https://github.com/diegotita4/credit-approval-model                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# IMPORT NECESSARY LIBRARIES
import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.metrics import classification_report, roc_auc_score, matthews_corrcoef, f1_score, confusion_matrix, auc, roc_curve

# IMPORT NECESSARY MODULES FROM OTHER SCRIPTS
from eda import ExploratoryDataAnalysis

# --------------------------------------------------

class CreditCardModel:
    """
    A class to construct a credit card approval model, capable of preparing data, training different types of models,
    and evaluating their performance.

    Attributes:
        data_path (str): Path to the dataset.
        target_column (str): Name of the target column in the dataset.
        selected_columns (list): Columns to be used in the model.
        selected_columns_to_drop (list): Columns to be excluded from the model.
        model_type (str): Type of model to train (e.g., 'logistic_regression', 'random_forest', 'xgboost', 'neural_network').

    Methods:
        eda(): Performs exploratory data analysis on the dataset.
        prepare_data(): Prepares the data for training by handling missing values and scaling features.
        build_model(): Constructs the model based on the specified type.
        train_model(): Trains the constructed model.
        evaluate_model(): Evaluates the model and prints out performance metrics.
        plot_metrics(): Generates plots for evaluation metrics such as ROC curve and confusion matrix.
        save_model(): Saves the trained model to a specified directory.
    """

    # ------------------------------
             
    def __init__(self, data_path, target_column, selected_columns, selected_columns_to_drop, model_type):
        """
        Initializes the CreditCardModel with specified attributes and data preparation utilities like scaler and imputer.
        """

        print(f"\n{'='*50}\nInitializing 'CreditModel' ({model_type})...\n{'='*50}")

        self.data = pd.read_excel(data_path)[selected_columns]
        self.target_column = target_column
        self.selected_columns = selected_columns
        self.selected_columns_to_drop = selected_columns_to_drop
        self.model_type = model_type
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')

    # ------------------------------

    def eda(self, data_path):
        """
        Performs exploratory data analysis by utilizing the ExploratoryDataAnalysis class on the selected columns of the data.
        """

        print(f"\n{'='*50}\nPerforming 'Exploratory Data Analysis' (EDA)...\n{'='*50}")

        data = pd.read_excel(data_path)[self.selected_columns]
        eda_instance = ExploratoryDataAnalysis(data, self.target_column)
        eda_instance.perform_EDA()

    # ------------------------------

    def prepare_data(self):
        """
        Prepares the data for training by handling missing values, encoding categorical variables, scaling features,
        and splitting the data into training and testing sets.
        """

        print(f"\n{'='*50}\nPreparing data for training...\n{'='*50}")

        try:
            features = [col for col in self.data.columns if col not in self.selected_columns_to_drop and col != self.target_column]
            X = self.data[features]
            y = self.data[self.target_column]
            unique_classes = y.unique()

            if len(unique_classes) == 2:
                y = y.map({unique_classes[0]: 0, unique_classes[1]: 1})
                print(f"\nClasses detected ({len(unique_classes)}) in 'target_column': '{self.target_column}' -> '{unique_classes[0]}' = 0 / '{unique_classes[1]}' = 1.")

            else:
                raise ValueError(f"\nTarget column: '{self.target_column}' must have exactly two unique classes. Found: {unique_classes}.")

            X_imputed = self.imputer.fit_transform(X)
            X_scaled = self.scaler.fit_transform(X_imputed)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        except Exception as e:
            print(f"Error in preparing data: {e}")

    # ------------------------------

    def build_model(self):
        """
        Constructs the machine learning model based on the specified type in the model_type attribute.
        """

        print(f"\n{'='*50}\nBuilding model...\n{'='*50}")

        try:
            if self.model_type == 'logistic_regression':
                self.model = LogisticRegression(
                    penalty='elasticnet', 
                    dual=False, 
                    tol=0.0004, 
                    fit_intercept=True, 
                    solver='saga', 
                    max_iter=1500, 
                    l1_ratio=1, 
                    C=1
                )

            elif self.model_type == 'random_forest':
                self.model = RandomForestClassifier(n_estimators=100)

            elif self.model_type == 'xgboost':
                self.model = XGBClassifier(tree_method='hist', device='cpu')

            elif self.model_type == 'neural_network':
                input_layer = Input(shape=(self.X_train.shape[1],))
                x = Dense(32, activation='relu')(input_layer)
                x = Dropout(0.2)(x)
                x = Dense(16, activation='relu')(x)
                x = Dropout(0.1)(x)
                output_layer = Dense(1, activation='sigmoid')(x)
                self.model = Model(inputs=input_layer, outputs=output_layer)
                self.model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy', 'AUC'])

        except Exception as e:
            print(f"Error in building model: {e}")

    # ------------------------------

    def train_model(self, epochs=None, batch_size=None):
        """
        Trains the constructed model. For neural networks, it includes early stopping to avoid overfitting.
        """

        print(f"\n{'='*50}\nTraining model...\n{'='*50}")

        try:
            if self.model_type == 'neural_network':
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                return self.model.fit(self.X_train, self.y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=1)

            else:
                self.model.fit(self.X_train, self.y_train)
                return None

        except Exception as e:
            print(f"Error in training model: {e}")

    # ------------------------------

    def evaluate_model(self):
        """
        Evaluates the model using standard classification metrics and prints the results.
        """

        print(f"\n{'='*50}\nEvaluating model...\n{'='*50}")

        try:
            predictions = (self.model.predict(self.X_test) > 0.5).astype(int).flatten() if self.model_type == 'neural_network' else self.model.predict(self.X_test)
            probas = self.model.predict_proba(self.X_test)[:, 1] if not self.model_type == 'neural_network' else self.model.predict(self.X_test).flatten()
            print('\n* Classification Report:\n')
            print(classification_report(self.y_test, predictions))
            print(f"{'-'*50}\n\n* Matthews Correlation Coefficient:", np.round(matthews_corrcoef(self.y_test, predictions), 6))
            print('* F1-Score:', np.round(f1_score(self.y_test, predictions), 6))
            print('* ROC AUC Score:', np.round(roc_auc_score(self.y_test, probas), 6))
            self.plot_metrics(predictions, probas)

        except Exception as e:
            print(f"Error in evaluating model: {e}")

    # ------------------------------

    def plot_metrics(self, predictions, probas):
        """
        Generates and displays plots for the evaluation metrics of the model, including ROC curves and confusion matrices.
        """

        print(f"\n{'='*50}\nPlotting evaluation metrics...\n{'='*50}")

        fpr, tpr, thresholds = roc_curve(self.y_test, probas)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()

        cm = confusion_matrix(self.y_test, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    # ------------------------------

    def save_model(self, model_directory='models'):
        """
        Saves the trained model to a specified directory. The file format varies based on the model type.
        """

        print(f"\n{'='*50}\nSaving model...\n{'='*50}")

        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

            print(f"\n¡Created directory: '{model_directory}'!\n\n{'-'*50}")

        if self.model_type == 'neural_network':
            filename = f'{model_directory}/{self.model_type}_model.keras'
            self.model.save(filename)

        else:
            filename = f'{model_directory}/{self.model_type}_model.pkl'
            joblib.dump(self.model, filename)

        print(f"\n¡Model saved: '{filename}'!")
