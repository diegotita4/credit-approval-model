# Credit Approval Model

Binary classification model that predicts whether a credit application should be approved, based on four financial variables of the applicant. Built and compared with logistic regression, random forest, XGBoost and a neural network.

Credit Models course · ITESO (FlipBank case) · Aug – Dec 2024 · Team of 5

## Problem

A lender needs a consistent, data-driven rule to approve or reject applications. The dataset has ~92k applications and is highly imbalanced between approved and rejected cases, so accuracy alone is misleading and models are compared by ROC AUC.

## Approach

1. **Data quality review** (`dqr.py`): validation, outlier handling, optional imputation of missing values and encoding.
2. **Exploratory analysis** (`eda.py`): summaries, distributions, box plots, correlations and bivariate plots.
3. **Modeling** (`model.py`): four features (estimated income, expenses, payment capacity and indebtedness), standard scaling and an 80/20 train/test split. Four model types are available: logistic regression, random forest, XGBoost and a neural network (Keras).
4. **Evaluation:** classification report, ROC AUC, F1 and Matthews correlation coefficient, plus ROC curve and confusion matrix.

## Results

The logistic regression reached a **ROC AUC of 0.78 on ~18k hold-out applications**, benchmarked against random forest, XGBoost and a neural network.

## Limitations

- No resampling or class weights were applied to handle the imbalance; that is the main next step.
- The imputer and scaler are fitted before the train/test split, which can leak a little information into the test set.
- The model uses only four variables.

## Data

The applicant data used in the course is not included. To run the pipeline, place an Excel file at `data/data.xlsx` with the columns `Ingreso_Estimado`, `Egresos`, `Capacidad_Pago`, `Endeudamiento` and the target `ESTATUS` (two classes).

## How to run

```bash
git clone https://github.com/diegotita4/credit-approval-model.git
cd credit-approval-model
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Choose the model with `model_type` in `main.py`: `logistic_regression`, `random_forest`, `xgboost` or `neural_network`. Trained models are saved to `models/`.

## Team

Juan Antonio Mugica Liparoli ([@Antonio-IF](https://github.com/Antonio-IF)) · Ana Sofía Brizuela Casarín ([@anasofiabrizuela](https://github.com/anasofiabrizuela)) · Diego Emilio Enriquez Nares ([@diegotita4](https://github.com/diegotita4)) · Luis Robles ([@luisrc44](https://github.com/luisrc44)) · Óscar Uriel Alvarado Garnica ([@Oscar148](https://github.com/Oscar148))

Originally developed in [@Antonio-IF](https://github.com/Antonio-IF)'s repository and republished here without data or internal documents.

## License

MIT License © 2024 Juan Antonio Mugica Liparoli
