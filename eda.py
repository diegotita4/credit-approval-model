
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: CREDIT CARD MODEL                                                                          -- #
# -- script: eda.py : python script with EDA functionality                                               -- #
# -- author: anasofiabrizuela / Antonio-IF / diegotita4 / luisrc44 / Oscar148                            -- #
# -- license: MIT License                                                                                -- #
# -- repository: https://github.com/diegotita4/credit-approval-model                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# IMPORT NECESSARY LIBRARIES
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# --------------------------------------------------

class ExploratoryDataAnalysis:
    """
    Class for performing Exploratory Data Analysis on a given dataset. It provides methods to
    summarize, visualize, and understand the underlying patterns of the data.

    Attributes:
        data (DataFrame): The dataset on which EDA will be performed.
        target_column (str): The name of the target column in the dataset.
        number_columns (Index): Column names of numeric columns in the dataset.
        object_columns (Index): Column names of object (categorical) columns in the dataset.

    Methods:
        validate_data(): Checks if the provided DataFrame is empty.
        data_summary(): Provides a summary of the dataset including head, tail, types, and more.
        data_statistics(): Calculates and prints out statistics like mean, median, mode, etc.
        setup_subplots(): Configures subplots for data visualization.
        plot_count_plots(): Plots count plots for categorical data.
        plot_histograms(): Plots histograms for numerical data.
        plot_box_plots(): Plots box plots for numerical data to check for outliers.
        plot_correlation_matrix(): Plots a correlation matrix for numerical data.
        plot_histograms_bivariates(): Plots histograms for bivariate analysis.
        plot_violins(): Plots violin plots for numerical data against the target variable.
        perform_EDA(): Executes all EDA methods systematically.
    """

    # ------------------------------

    def __init__(self, data, target_column):
        """
        Initializes the ExploratoryDataAnalysis class with the dataset and target column.
        Also identifies numerical and categorical columns from the data.
        """

        self.validate_data(data)
        self.data = data
        self.target_column = target_column
        self.number_columns = self.data.select_dtypes(include=['number']).columns
        self.object_columns = self.data.select_dtypes(include=['object']).columns

    # ------------------------------

    def validate_data(self, data):
        """
        Validates that the data is not empty, raises ValueError if it is.
        """

        if data.empty:
            raise ValueError('\nDataFrame must not be empty.')

    # ------------------------------

    def data_summary(self):
        """
        Prints a detailed summary of the dataset including the first and last rows, shape,
        data types, unique values, and more.
        """

        print(f"\n{'='*50}\nData summary...\n{'='*50}")

        try:
            print(f'\n* Head:\n')
            print(self.data.head(5))

            print(f"\n{'-'*50}\n\n* Tail:\n")
            print(self.data.tail(5))

            print(f"\n{'-'*50}\n\n* Shape:\n")
            print(self.data.shape)

            print(f"\n{'-'*50}\n\n* Columns:\n")
            print(len(self.data.columns))
            print(self.data.columns)

            print(f"\n{'-'*50}\n\n* Data types:\n")
            print(self.data.dtypes)

            print(f"\n{'-'*50}\n\n* Numeric columns:\n")
            print(len(self.number_columns))
            print(self.number_columns)

            print(f"\n{'-'*50}\n\n* Object columns:\n")
            print(len(self.object_columns))
            print(self.object_columns)

            print(f"\n{'-'*50}\n\n* Unique values:\n")
            for col in self.data.columns:
                unique_values = self.data[col].nunique()

                print(f'{col}: {unique_values}')

                if unique_values < 20:
                    print(f"\nLabels in '{col}': {sorted(self.data[col].unique())}")

            print(f"\n{'-'*50}\n\n* Information:\n")
            print(self.data.info())

            print(f"\n{'-'*50}\n\n* Describe:\n")
            print(self.data.describe())

            print(f"\n{'-'*50}\n\n* Missing values:\n")
            print(self.data.isnull().sum())

            print(f"\n{'-'*50}\n\n* Outliers:")
            for col in self.number_columns:
                if self.data[col].dtype in ['int64', 'float64']:
                    z_scores = np.abs(stats.zscore(self.data[col].dropna()))

                    print(f"\n   - Outliers in '{col}': {np.where(z_scores > 3)}")
    
        except Exception as e:
            print(f'\nError during the data summary: {e}')

    # ------------------------------

    def data_statistics(self):
        """
        Prints descriptive statistics for numerical columns in the dataset.
        """

        print(f"\n{'='*50}\nData statistics...\n{'='*50}")

        try:
            numeric_data = self.data[self.number_columns]

            print(f'\n* Mean:\n')
            print(numeric_data.mean())

            print(f"\n{'-'*50}\n\n* Median:\n")
            print(numeric_data.median())

            print(f"\n{'-'*50}\n\n* Mode:\n")
            print(numeric_data.mode().iloc[0])

            print(f"\n{'-'*50}\n\n* Minimum:\n")
            print(numeric_data.min())

            print(f"\n{'-'*50}\n\n* Maximum:\n")
            print(numeric_data.max())

            print(f"\n{'-'*50}\n\n* Variance:\n")
            print(numeric_data.var())

            print(f"\n{'-'*50}\n\n* Standard deviation:\n")
            print(numeric_data.std())

        except Exception as e:
            print(f'\nError during the data statistics: {e}')

    # ------------------------------

    def setup_subplots(self, data_columns, plot_type, title_prefix, hue=None, anaysis=None):
        """
        Configures and displays subplots for different types of visualizations specified by the plot_type parameter.
        """

        n_rows = (len(data_columns) + 3) // 4

        if n_rows == 0:
            print('\nNo rows to plot.')
            return
    
        fig, axs = plt.subplots(n_rows, 4, figsize=(20, n_rows * 5))
        axs = axs.flatten()

        for i, col in enumerate(data_columns):
            try:
                ax = axs[i]

                if plot_type == 'hist':
                    sns.histplot(self.data[col], bins=30, kde=True, ax=ax, color='skyblue')
                
                elif plot_type == 'box':
                    sns.boxplot(x=self.data[col], ax=ax, color='lightgreen')
                
                elif plot_type == 'count':
                    if anaysis == 'yes':
                        sns.countplot(x=col, data=self.data, ax=ax, hue=hue, palette='viridis', legend=False)

                    else:
                        sns.countplot(x=col, data=self.data, ax=ax, hue=self.data[col], palette='viridis', legend=False)

                elif plot_type == 'violin':
                    sns.violinplot(x=self.target_column, y=self.data[col], data=self.data, ax=ax, palette='pastel', hue=self.target_column)

                ax.set_title(f'{title_prefix} {col}', fontsize=14)
                ax.set_xlabel('')
                ax.set_ylabel('Count' if plot_type == 'count' else 'Value')

            except Exception as e:
                print(f'\nError plotting {col}: {e}')

        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.show()

    # ------------------------------

    def plot_count_plots(self):
        """
        Plots count plots for all categorical columns with less than 20 unique values.
        """

        print(f"\n{'='*50}\nCountplot...\n{'='*50}")

        filtered_object_columns = [col for col in self.object_columns if self.data[col].nunique() < 20]
        self.setup_subplots(filtered_object_columns, 'count', 'Count of')

    # ------------------------------

    def plot_histograms(self):
        """
        Plots histograms for all numerical columns to visualize distributions.
        """

        print(f"\n{'='*50}\nHistogram...\n{'='*50}")

        self.setup_subplots(self.number_columns, 'hist', 'Distribution of')

    # ------------------------------

    def plot_box_plots(self):
        """
        Plots box plots for all numerical columns to identify outliers.
        """

        print(f"\n{'='*50}\nBoxplot...\n{'='*50}")

        self.setup_subplots(self.number_columns, 'box', 'Box plot of')

    # ------------------------------

    def plot_correlation_matrix(self):
        """
        Displays a heatmap of correlations between numerical columns in the dataset.
        """

        print(f"\n{'='*50}\nCorrelation matrix...\n{'='*50}")

        try:
            numeric_data = self.data[self.number_columns]

            plt.figure(figsize=(12, 8))
            sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f')
            plt.title('Correlation Matrix:', fontsize=16)
            plt.show()

        except Exception as e:
            print(f'\nError plotting correlation matrix: {e}')

    # ------------------------------

    def plot_histograms_bivariates(self):
        """
        Plots bivariate histograms comparing each categorical column with the target variable.
        """

        print(f"\n{'='*50}\nBivariate histogram...\n{'='*50}")

        filtered_object_columns = [col for col in self.object_columns if col != self.target_column and self.data[col].nunique() < 20]

        self.setup_subplots(filtered_object_columns, 'count', f"'{self.target_column}' VS", f"'{self.target_column}'", 'yes')

    # ------------------------------

    def plot_violins(self):
        """
        Plots violin plots to compare the distribution of numerical data against the target column.
        """

        print(f"\n{'='*50}\nViolin...\n{'='*50}")

        self.setup_subplots(self.number_columns, 'violin', f"'{self.target_column}' VS", f"'{self.target_column}'")

    # ------------------------------

    def perform_EDA(self):
        """
        Executes a comprehensive EDA by running all analysis and plotting functions.
        """

        self.data_summary()
        self.data_statistics()
        self.plot_count_plots()
        self.plot_histograms()
        self.plot_box_plots()
        self.plot_correlation_matrix()
        self.plot_histograms_bivariates()
        self.plot_violins()
