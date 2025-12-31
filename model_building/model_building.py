# ==============================
# IMPORT REQUIRED LIBRARIES
# ==============================

import numpy as np
import pandas as pd

# Feature selection
from sklearn.feature_selection import chi2

# Preprocessing & pipelines
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    LabelEncoder,
    OrdinalEncoder,
    MinMaxScaler
)

# Handling class imbalance
from imblearn.over_sampling import SMOTE

# Model training & evaluation
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Utility
import pickle
import warnings
warnings.filterwarnings("ignore")


# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv("Traveling_Data.csv")
df.head()

# Check class distribution (imbalanced dataset)
df.ProdTaken.value_counts()


# ==============================
# CHI-SQUARE TEST (FEATURE SELECTION)
# ==============================
# Chi-square test checks dependency between categorical features and target

le = LabelEncoder()

# TypeofContact
TypeofContact_encoded = le.fit_transform(df["TypeofContact"])
chi_scores, p_values = chi2(TypeofContact_encoded.reshape(-1, 1), df["ProdTaken"])
p_values

# CityTier
chi_scores, p_values = chi2(df["CityTier"].values.reshape(-1, 1), df["ProdTaken"])
p_values

# Occupation
Occupation_encoded = le.fit_transform(df["Occupation"])
chi_scores, p_values = chi2(Occupation_encoded.reshape(-1, 1), df["ProdTaken"])
p_values

# Gender
Gender_encoded = le.fit_transform(df["Gender"])
chi_scores, p_values = chi2(Gender_encoded.reshape(-1, 1), df["ProdTaken"])
p_values

# NumberOfPersonVisiting
NumberOfPersonVisiting_encoded = le.fit_transform(df["NumberOfPersonVisiting"])
chi_scores, p_values = chi2(NumberOfPersonVisiting_encoded.reshape(-1, 1), df["ProdTaken"])
p_values

# NumberOfFollowups
NumberOfFollowups_encoded = le.fit_transform(df["NumberOfFollowups"])
chi_scores, p_values = chi2(NumberOfFollowups_encoded.reshape(-1, 1), df["ProdTaken"])
p_values

# ProductPitched
ProductPitched_encoded = le.fit_transform(df["ProductPitched"])
chi_scores, p_values = chi2(ProductPitched_encoded.reshape(-1, 1), df["ProdTaken"])
p_values

# PreferredPropertyStar
chi_scores, p_values = chi2(df["PreferredPropertyStar"].values.reshape(-1, 1), df["ProdTaken"])
p_values

# MaritalStatus
MaritalStatus_encoded = le.fit_transform(df["MaritalStatus"])
chi_scores, p_values = chi2(MaritalStatus_encoded.reshape(-1, 1), df["ProdTaken"])
p_values

# NumberOfTrips
chi_scores, p_values = chi2(df["NumberOfTrips"].values.reshape(-1, 1), df["ProdTaken"])
p_values

# Passport
chi_scores, p_values = chi2(df["Passport"].values.reshape(-1, 1), df["ProdTaken"])
p_values

# PitchSatisfactionScore
chi_scores, p_values = chi2(df["PitchSatisfactionScore"].values.reshape(-1, 1), df["ProdTaken"])
p_values

# OwnCar
chi_scores, p_values = chi2(df["OwnCar"].values.reshape(-1, 1), df["ProdTaken"])
p_values

# NumberOfChildrenVisiting
chi_scores, p_values = chi2(df["NumberOfChildrenVisiting"].values.reshape(-1, 1), df["ProdTaken"])
p_values

# Designation
Designation_encoded = le.fit_transform(df["Designation"])
chi_scores, p_values = chi2(Designation_encoded.reshape(-1, 1), df["ProdTaken"])
p_values


# ==============================
# FEATURE SELECTION
# ==============================

X = df[
    [
        "Age",
        "CityTier",
        "NumberOfFollowups",
        "ProductPitched",
        "PreferredPropertyStar",
        "MaritalStatus",
        "Passport",
        "PitchSatisfactionScore",
        "Designation"
    ]
]

y = df["ProdTaken"]


# ==============================
# DATA CLEANING
# ==============================

# Convert Passport from 0/1 to categorical Yes/No
df["Passport"] = (
    df["Passport"]
    .astype(str)
    .str.replace("0", "No")
    .str.replace("1", "Yes")
)

df["Passport"].value_counts()

# Basic checks
X.Age.min(), X.Age.max()
X.Designation.value_counts()


# ==============================
# TRAIN-TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.7,
    random_state=42
)


# ==============================
# PREPROCESSING PIPELINE
# ==============================

preprocessor = ColumnTransformer(
    transformers=[
        ("age_scaled", MinMaxScaler(), ["Age"]),
        (
            "product_pitched_encoded",
            OrdinalEncoder(
                categories=[["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]]
            ),
            ["ProductPitched"]
        ),
        ("marital_status_encoded", OneHotEncoder(handle_unknown="ignore"), ["MaritalStatus"]),
        ("passport_encoded", OneHotEncoder(handle_unknown="ignore"), ["Passport"]),
        ("designation_encoded", OneHotEncoder(handle_unknown="ignore"), ["Designation"]),
    ],
    remainder="passthrough"
)

# Transform data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)


# ==============================
# HANDLE CLASS IMBALANCE (SMOTE)
# ==============================

smote = SMOTE(random_state=42)

Xtrain_resampled_smote, ytrain_resampled_smote = smote.fit_resample(
    X_train_transformed, y_train
)

Xtest_resampled_smote, ytest_resampled_smote = smote.fit_resample(
    X_test_transformed, y_test
)

ytrain_resampled_smote.value_counts()


# ==============================
# MODEL COMPARISON
# ==============================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(Xtrain_resampled_smote, ytrain_resampled_smote)
    y_pred = model.predict(Xtest_resampled_smote)

    acc = accuracy_score(ytest_resampled_smote, y_pred)
    results[name] = acc

    print(f"\n{name}")
    print("Accuracy:", acc)


# ==============================
# CROSS VALIDATION
# ==============================

model = RandomForestClassifier()

scores = cross_val_score(
    model,
    np.vstack([Xtrain_resampled_smote, Xtest_resampled_smote]),
    np.hstack([ytrain_resampled_smote, ytest_resampled_smote]),
    cv=5,
    scoring="accuracy"
)

print("Accuracy Scores:", scores)
print("Mean Accuracy:", np.mean(scores))


# ==============================
# HYPERPARAMETER TUNING
# ==============================

rf = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "min_samples_split": [2, 3, 4, 5],
    "min_samples_leaf": [1, 2, 3, 4, 5],
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(Xtrain_resampled_smote, ytrain_resampled_smote)

print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)


# ==============================
# FINAL MODEL TRAINING
# ==============================

model = RandomForestClassifier(
    random_state=42,
    max_depth=14,
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=100,
    class_weight="balanced"
)

model.fit(Xtrain_resampled_smote, ytrain_resampled_smote)

y_pred = model.predict(Xtest_resampled_smote)


# ==============================
# MODEL EVALUATION
# ==============================

accuracy = accuracy_score(ytest_resampled_smote, y_pred)
cm = confusion_matrix(ytest_resampled_smote, y_pred)
precision = precision_score(ytest_resampled_smote, y_pred)
recall = recall_score(ytest_resampled_smote, y_pred)
f1 = f1_score(ytest_resampled_smote, y_pred)

print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", cm)
print("\nPrecision:", precision)
print("\nRecall:", recall)
print("\nF1-Score:", f1)
print("\nClassification Report:\n")
print(classification_report(ytest_resampled_smote, y_pred))


# ==============================
# SAVE MODEL & PREPROCESSOR
# ==============================

with open("tourism_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

print("Model and Preprocessor saved successfully")


# ==============================
# CHECK SCIKIT-LEARN VERSION
# ==============================

import sklearn
print("Scikit-learn version:", sklearn.__version__)
