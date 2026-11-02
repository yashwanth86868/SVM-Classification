
# Support Vector Machine (SVM) Classification

A beginner-friendly machine learning project that trains a Support Vector Machine classifier with an RBF kernel, evaluates its performance, saves the trained model and scaler, and provides a Streamlit web interface for predictions.

## Project features

- Loads a CSV dataset using pandas
- Splits the data into training and testing sets
- Scales numerical features with `StandardScaler`
- Trains an SVM classifier using the RBF kernel
- Evaluates the model using accuracy, confusion matrix, and classification report
- Generates a confusion-matrix image
- Generates an ROC curve for binary classification
- Saves the trained model and scaler using pickle
- Provides a Streamlit frontend for interactive predictions

## Project structure

```text
SVM-Classification/
├── .github/
│   └── workflows/
│       └── python-ci.yml
├── app.py
├── svm_classification.py
├── logit classification.csv
├── svm_model.pkl
├── scaler.pkl
├── confusion_matrix.png
├── roc_curve.png
├── requirements.txt
├── .gitignore
└── README.md
```

> Keep the dataset filename and the filename used in `pd.read_csv()` consistent. If your dataset contains private or sensitive information, do not upload it to a public repository.

## Requirements

- Python 3.10 or newer
- pip

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR-USERNAME/SVM-Classification.git
   cd SVM-Classification
   ```
2. Create and activate a virtual environment:

   **Windows (PowerShell):**

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   **macOS / Linux:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Train the model

Make sure the CSV dataset is in the project folder. In `svm_classification.py`, update the CSV filename if needed. Avoid using a machine-specific absolute path such as `C:\Users\...`.

Run:

```bash
python svm_classification.py
```

The training script should create:

- `svm_model.pkl` — trained SVM model
- `scaler.pkl` — fitted feature scaler
- `confusion_matrix.png` — confusion matrix plot
- `roc_curve.png` — ROC curve (binary classification only)

The ROC curve code requires a binary classification target and a model configured with `probability=True`.

## Run the Streamlit application

Run model training first so that `svm_model.pkl` and `scaler.pkl` exist. Then:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit in your browser.

The app currently accepts two feature values. Update the labels and input ranges in `app.py` to match the two columns selected in the training script (`dataset.iloc[:, 2:4]`).

## Model evaluation

The training script reports:

- **Accuracy:** proportion of predictions that are correct
- **Confusion matrix:** counts of correct and incorrect predictions by class
- **Precision:** proportion of predicted positives that are correct
- **Recall:** proportion of actual positives that are identified
- **F1-score:** harmonic mean of precision and recall
- **Training and test scores:** model accuracy on each split; these are not direct measurements of bias and variance

## GitHub Actions

The workflow in `.github/workflows/python-ci.yml` runs automatically when code is pushed or a pull request is opened against the `main` branch. It installs dependencies and checks Python syntax for the project scripts.

The workflow intentionally does not run model training because the dataset may not be committed to GitHub. If you choose to include a suitable public sample dataset, you can extend the workflow to run training tests as well.

## Important notes

- The training and prediction code must use the same feature order.
- Always call `fit_transform()` on training data and `transform()` on test or user-input data.
- Do not commit secrets, private datasets, or unnecessary large files.
- Pickle files should only be loaded from sources you trust.
- `degree` is used by the polynomial SVM kernel; it does not affect an RBF-kernel model.

## Author

**Your Name**

Balija Yashwanth

[github.com/yashwanth86868](https://github.com/yashwanth86868)
