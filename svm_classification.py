
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Evaluation metrics
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# Save model
import pickle

# Load dataset
dataset = pd.read_csv(r"C:\Users\ybali\Downloads\logit classification.csv")

# Independent and dependent variables
X = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, -1].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=0
)

# Feature scaling
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Create SVM classifier
classifier = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,
    random_state=0
)

# Train model
classifier.fit(X_train, y_train)

# Predictions
y_pred = classifier.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Training accuracy
bias = classifier.score(X_train, y_train)
print("Training Accuracy:", bias)

# Testing accuracy
variance = classifier.score(X_test, y_test)
print("Testing Accuracy:", variance)

# Confusion matrix visualization
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("SVM Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# ROC Curve (binary classification only)
if len(np.unique(y)) == 2:
    y_prob = classifier.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("SVM ROC Curve")
    plt.legend()
    plt.savefig("roc_curve.png")
    plt.show()

# Save model and scaler
with open("svm_model.pkl", "wb") as file:
    pickle.dump(classifier, file)

with open("scaler.pkl", "wb") as file:
    pickle.dump(sc, file)

print("\nModel and scaler saved successfully!")
