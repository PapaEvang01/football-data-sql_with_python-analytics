from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# Football Match Prediction - ML Models
# ============================================================
#
# Description:
# This script defines the Machine Learning models used
# for football match prediction.
#
# The models are designed to classify:
# - match outcomes (W1 / D / W2)
# - Over/Under 2.5 goals
#
# Two different approaches are implemented:
# - Logistic Regression
# - Random Forest Classifier
#
# The purpose is to compare:
# - linear vs non-linear models
# - interpretability vs complexity
# - statistical vs ensemble-based learning
# ============================================================


# ============================================================
# LOGISTIC REGRESSION MODEL
# ============================================================
# Pipeline-based model including:
# - feature normalization
# - multiclass logistic regression classifier

def build_logistic_regression():

    model = Pipeline([

        # Standardize feature values before training
        ("scaler", StandardScaler()),

        # Multiclass classification model
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
            multi_class="auto"
        ))
    ])

    return model


# ============================================================
# RANDOM FOREST MODEL
# ============================================================
# Ensemble-based classification model using multiple
# decision trees.
#
# The model is configured to reduce overfitting while
# maintaining predictive performance.

def build_random_forest():

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced"
    )

    return model


# ============================================================
# MODEL COLLECTION
# ============================================================
# Store all available models inside a dictionary for
# easier experimentation and evaluation.

def get_models():

    models = {
        "logistic_regression": build_logistic_regression(),
        "random_forest": build_random_forest()
    }

    return models