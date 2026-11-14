# IEEE-CIS Fraud Detection

## 1. Project Overview

This project develops and evaluates machine learning models for detecting fraudulent transactions using the IEEE-CIS Fraud Detection dataset.

The main objective is to identify potentially fraudulent transactions in a highly imbalanced dataset, where fraudulent transactions account for only a small proportion of all transactions.

The project covers the complete machine learning workflow, including:

- Data integration and preprocessing.
- Exploratory Data Analysis (EDA).
- Feature engineering.
- Feature selection.
- Model development and evaluation.
- Model comparison and performance analysis.

Five classification models are evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- LightGBM
- XGBoost

A Dummy Classifier is also used as a baseline for comparison.

## 2. Repository Structure

```text
IEEE-CIS-Fraud-Detection/
│
├── notebooks/
│   ├── IEEE_CIS_Project_END.ipynb
│   └── README.md
│
├── code/
│   ├── step0_setup_and_split.py
│   ├── step1_advanced_eda.py
│   ├── step2_feature_engineering.py
│   └── step3_4_selection_and_handover.py
│
├── outputs/
│   └── README.md
│
├── artifacts/
│   └── README.md
│
├── README.md
├── requirements.txt
└── .gitignore
```

**Directory descriptions:**

| Directory / File | Description |
|---|---|
| `notebooks/` | Main end-to-end Jupyter Notebook containing the complete project workflow. |
| `code/` | Python scripts used for data preparation, EDA, feature engineering, and feature selection. |
| `outputs/` | Exported evaluation results, tables, and visualization plots. |
| `artifacts/` | Supporting files generated during the machine learning workflow, such as selected feature configurations. |
| `requirements.txt` | Python package dependencies. |
| `.gitignore` | Specifies files and directories excluded from Git tracking. |

## 3. Dataset

This project uses the **IEEE-CIS Fraud Detection** dataset provided by Vesta Corporation through Kaggle.

The original dataset contains the following files:

- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`
- `sample_submission.csv`

The dataset is not included in this repository because of its size and distribution restrictions.

### Download the Dataset

1. Visit the [IEEE-CIS Fraud Detection competition on Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data).
2. Download the dataset files.
33.Extract the downloaded files.

**Important:** Place the extracted CSV files in the `data/` directory when preparing the project environment.

## 4. How to Run the Project

The recommended way to run this project is **Google Colab**. The main notebook contains the end-to-end workflow and is designed to be executed sequentially.

### Step 1. Prepare the Dataset

Download and extract the IEEE-CIS dataset from Kaggle as described in Section 3.

*IMPORTANT* : Create the following directory structure in Google Drive:


MyDrive/
└── IEEE_CIS_Project/
    │
    ├── data/
    │   ├── train_transaction.csv
    │   ├── train_identity.csv
    │   ├── test_transaction.csv
    │   ├── test_identity.csv
    │   └── sample_submission.csv
    │
    ├── code/
    │   ├── step0_setup_and_split.py
    │   ├── step1_advanced_eda.py
    │   ├── step2_feature_engineering.py
    │   └── step3_4_selection_and_handover.py
    │
    ├── outputs/
    │
    └── artifacts/
```

Upload the dataset CSV files into `data/` and the Python scripts into `code/`.

### Step 2. Open the Notebook in Google Colab

1. Open the `notebooks/` directory in this GitHub repository.
2. Select `IEEE_CIS_Project_END.ipynb`.
3. Download the notebook using the GitHub download option.
4. Open [Google Colab](https://colab.research.google.com/).
5. Select **File → Upload notebook** and upload the downloaded `.ipynb` file.

Alternatively, download the notebook first and open it directly in Google Colab.

### Step 3. Connect Google Drive

In Google Colab:

1. Open the notebook.
2. Run the Google Drive mounting cell.
3. Authorize access to your Google Drive.
4. Make sure the project directory is available at:

```python
/content/drive/MyDrive/IEEE_CIS_Project
```

If your project is stored in a different location, update the project path in the notebook accordingly.

### Step 4. Install Dependencies

Install the required Python packages in the Colab environment.

If the notebook does not already contain a dependency installation cell, run:

```python
!pip install pandas numpy scikit-learn joblib lightgbm xgboost matplotlib seaborn
```

### Step 5. Execute the Notebook

Run the notebook from the first cell to the last cell in the original order.

In Google Colab, select:

**Runtime → Run all**

The notebook performs the following stages:

1. Environment setup and data loading.
2. Data integration and initial preprocessing.
3. Exploratory Data Analysis.
4. Feature engineering.
5. Feature selection.
6. Data preprocessing for different model families.
7. Model training and validation.
8. Model evaluation and comparison.
9. Exporting results and visualizations.

**Important notes:**

- Do not skip cells unless you understand their dependencies.
- Keep the required dataset files and project folders in their expected locations.
- Some stages may take considerable time because of the dataset size and model training process.
- Make sure the Colab runtime has sufficient memory and computational resources.

### Step 6. Access the Results

After successful execution, generated files are saved in the project directories, mainly:

- `outputs/`: model comparison tables, confusion matrices, ROC and Precision-Recall curves, threshold analysis, and feature importance visualizations.
- `artifacts/`: supporting files and configurations generated during preprocessing and feature selection.

Some large trained model files are not included in this GitHub repository. They are stored separately in the project environment.

## 5. Model Evaluation

The models are evaluated using a chronological training-validation split to preserve the temporal order of transactions.

The evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC (Average Precision)

Because the dataset is highly imbalanced, Precision, Recall, F1-score, and PR-AUC are particularly important when assessing fraud detection performance.

The experimental results indicate that XGBoost achieves the highest PR-AUC among the evaluated classification models, with LightGBM performing similarly.

## 6. Reproducibility

To reproduce the experiments:

1. Download the original dataset from Kaggle.
2. Prepare the required Google Drive directory structure.
3. Open the main notebook in Google Colab.
4. Install the required dependencies.
5. Execute all notebook cells sequentially.

The notebook provides the main implementation of the project, while the `code/`, `outputs/`, and `artifacts/` directories contain supporting scripts and generated results.

## 7. Notes

- The original dataset is not included in this repository.
- Large intermediate files and trained model files may be stored separately to avoid unnecessary repository size.
- Results may vary slightly depending on the runtime environment and computational resources.

## 8. Authors

This project was developed as part of an academic machine learning project on credit card transaction fraud detection using the IEEE-CIS dataset.
