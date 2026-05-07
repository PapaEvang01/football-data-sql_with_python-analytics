import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from ml_1_models import get_models


# ============================================================
# Football Match Prediction - ML Training Script
# ============================================================
#
# Description:
# This script trains and evaluates Machine Learning models
# for football match prediction tasks.
#
# The models are trained using the ML-ready dataset created
# in ML_dataset.py and focus on predicting:
# - match outcomes (W1 / D / W2)
# - Over/Under 2.5 goals
#
# The workflow includes:
# - dataset loading
# - feature selection
# - train/test splitting
# - model training
# - evaluation
# - prediction analysis
# ============================================================


# ============================================================
# FILE PATHS
# ============================================================
# Path to the ML-ready dataset and output directory
# used for storing results and predictions.

DATASET_PATH = Path("data/ml_ready/ml_1_big5_match_outcome_dataset.csv")
RESULTS_DIR = Path("results/ml_1")


# ============================================================
# FEATURE COLUMNS
# ============================================================
# Numerical features used as model inputs.
#
# The features combine:
# - team-level statistics
# - comparative difference metrics
# - historical matchup behavior

FEATURE_COLUMNS = [

    # Team performance metrics
    "team_1_win_rate_pct",
    "team_2_win_rate_pct",

    "team_1_avg_goal_difference",
    "team_2_avg_goal_difference",

    "team_1_avg_goals_scored",
    "team_2_avg_goals_scored",

    # Comparative strength features
    "diff_win_rate",
    "diff_goal_diff",
    "diff_goals_scored",

    # Matchup-level features
    "matchup_avg_goals",
    "matchup_intensity_score",
    "matchup_dominance"
]

## ============================================================
# MODEL TASK CONFIGURATION
# ============================================================
# Define the prediction task assigned to each model.
#
# Each task includes:
# - target column
# - class labels
# - descriptive task name

MODEL_TASKS = {

    # Multiclass football match outcome prediction
    "logistic_regression": {
        "target_column": "target_result",
        "labels": ["W1", "D", "W2"],
        "task_name": "match_outcome_prediction"
    },

    # Binary Over/Under 2.5 goals prediction
    "random_forest": {
        "target_column": "target_over_2_5",
        "labels": ["Under", "Over"],
        "task_name": "over_under_2_5_prediction"
    }
}


# ============================================================
# LOAD ML DATASET
# ============================================================
# Load the final Machine Learning-ready dataset
# generated from ML_dataset.py.

def load_ml_dataset():

    df = pd.read_csv(DATASET_PATH)

    print("Dataset loaded successfully.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return df
# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================
# Select model input features and the target variable.
# Rows with missing values in the selected columns are removed
# to ensure clean model training.

def prepare_features_and_target(df, target_column):

    # Remove rows with missing feature or target values
    df = df.dropna(subset=FEATURE_COLUMNS + [target_column]).copy()

    # Input features
    X = df[FEATURE_COLUMNS]

    # Prediction target
    y = df[target_column]

    print("\nFeatures and target prepared.")
    print(f"Target column: {target_column}")
    print(f"Final rows used: {len(df)}")

    print("\nTarget distribution:")
    print(y.value_counts())

    return X, y


# ============================================================
# SPLIT DATASET
# ============================================================
# Split the dataset into training and testing sets.
#
# Stratification is used to preserve the class distribution
# of the target variable in both sets.

def split_dataset(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    print("\nDataset split completed.")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    return X_train, X_test, y_train, y_test


# ============================================================
# TRAIN SINGLE MODEL
# ============================================================
# Fit one Machine Learning model using the training data.

def train_model(model, X_train, y_train):

    model.fit(X_train, y_train)

    return model

# ============================================================
# EVALUATE MODEL AND SAVE RESULTS
# ============================================================
# Evaluate a trained model on the test set and save the main
# performance outputs:
# - accuracy score
# - classification report
# - confusion matrix

def evaluate_model(model, X_test, y_test, model_name, labels):

    # Create a separate results folder for each model
    model_dir = RESULTS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Generate predictions on the test set
    y_pred = model.predict(X_test)

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, labels=labels)
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    print("\n" + "=" * 60)
    print(f"Evaluation results: {model_name}")
    print("=" * 60)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(report)

    print("\nConfusion Matrix:")
    print(matrix)

    # Save accuracy score
    with open(model_dir / "accuracy.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}")

    # Save full classification report
    with open(model_dir / "classification_report.txt", "w") as f:
        f.write(report)

    # Convert confusion matrix into a readable DataFrame
    cm_df = pd.DataFrame(
        matrix,
        index=[f"Actual_{label}" for label in labels],
        columns=[f"Pred_{label}" for label in labels]
    )

    # Save confusion matrix as CSV
    cm_df.to_csv(model_dir / "confusion_matrix.csv")


# ============================================================
# RUN FULL TRAINING PIPELINE
# ============================================================
# Run the complete training and evaluation process for all
# models defined in ML_models.py.
#
# Each model is connected to its own prediction task through
# the MODEL_TASKS configuration.

def run_training_pipeline(df):

    # Load all model definitions
    models = get_models()

    # Train and evaluate each model
    for model_name, model in models.items():

        # Get task-specific configuration
        task_config = MODEL_TASKS[model_name]

        target_column = task_config["target_column"]
        labels = task_config["labels"]

        print("\n" + "#" * 60)
        print(f"Running model: {model_name}")
        print(f"Target: {target_column}")
        print("#" * 60)

        # Prepare data for the selected target
        X, y = prepare_features_and_target(df, target_column)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = split_dataset(X, y)

        # Train model
        trained_model = train_model(model, X_train, y_train)

        # Evaluate model and save metrics
        evaluate_model(
            trained_model,
            X_test,
            y_test,
            model_name,
            labels
        )

        # Save prediction probabilities for further analysis
        save_prediction_probabilities(
            trained_model,
            X_test,
            y_test,
            df,
            model_name,
            labels,
            target_column
        )

# ============================================================
# SAVE PREDICTION PROBABILITIES
# ============================================================
# Save model predictions together with class probabilities.
#
# This output is useful because it shows not only the predicted
# class, but also how confident the model is for each possible
# outcome.

def save_prediction_probabilities(model, X_test, y_test, df, model_name, labels, target_column):

    # Create model-specific results folder
    model_dir = RESULTS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Keep match identifiers and the true target value
    test_df = df.loc[X_test.index, ["team_1", "team_2", target_column]].copy()

    # Generate class probabilities and final class predictions
    probabilities = model.predict_proba(X_test)
    predictions = model.predict(X_test)

    # Store probabilities in a readable DataFrame
    prob_df = pd.DataFrame(
        probabilities,
        columns=[f"{label}_prob" for label in model.classes_],
        index=X_test.index
    ).round(2)

    # Combine match information with prediction probabilities
    output_df = pd.concat([test_df, prob_df], axis=1)

    # Add final predicted class
    output_df["predicted_result"] = predictions

    # Keep one row per ordered matchup.
    # Example:
    # Arsenal vs Liverpool is treated differently from
    # Liverpool vs Arsenal because team order represents
    # match direction.
    output_df = output_df.drop_duplicates(
        subset=["team_1", "team_2"],
        keep="first"
    )

    # Sort output for cleaner presentation
    output_df = output_df.sort_values(
        by=["team_1", "team_2"]
    ).reset_index(drop=True)

    # Save prediction probabilities
    output_df.to_csv(model_dir / "prediction_probabilities.csv", index=False)

    print(f"Prediction probabilities saved for {model_name}.")


# ============================================================
# COMBINE MODEL OUTPUTS
# ============================================================
# Combine the outputs from both prediction tasks into one
# final table:
# - match outcome prediction
# - Over/Under 2.5 goals prediction

def create_final_combined_output():

    # Paths to individual model prediction outputs
    logreg_path = RESULTS_DIR / "logistic_regression" / "prediction_probabilities.csv"
    rf_path = RESULTS_DIR / "random_forest" / "prediction_probabilities.csv"

    # Load model prediction outputs
    logreg_df = pd.read_csv(logreg_path)
    rf_df = pd.read_csv(rf_path)

    # Rename prediction columns for clarity
    logreg_df = logreg_df.rename(columns={
        "predicted_result": "predicted_match_result"
    })

    rf_df = rf_df.rename(columns={
        "predicted_result": "predicted_over_under"
    })

    # Keep only relevant match outcome prediction columns
    logreg_df = logreg_df[
        ["team_1", "team_2", "W1_prob", "D_prob", "W2_prob", "predicted_match_result"]
    ]

    # Keep only relevant Over/Under prediction columns
    rf_df = rf_df[
        ["team_1", "team_2", "Under_prob", "Over_prob", "predicted_over_under"]
    ]

    # Merge both prediction outputs into one final table
    final_df = pd.merge(
        logreg_df,
        rf_df,
        on=["team_1", "team_2"],
        how="inner"
    )

    # Confidence score based on the highest match outcome probability
    final_df["confidence"] = final_df[["W1_prob", "D_prob", "W2_prob"]].max(axis=1).round(2)

    # Sort predictions by confidence from highest to lowest
    final_df = final_df.sort_values(
        by="confidence",
        ascending=False
    ).reset_index(drop=True)

    # Save final combined prediction table
    final_df.to_csv(RESULTS_DIR / "final_combined_predictions.csv", index=False)

    print("\nFinal combined prediction table created.")
    print(final_df.head())


# ============================================================
# MAIN
# ============================================================
# Main execution pipeline:
# 1. Create results folder
# 2. Load ML-ready dataset
# 3. Train and evaluate models
# 4. Save prediction probabilities
# 5. Create final combined prediction output

def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load ML-ready dataset
    df = load_ml_dataset()

    # Train models, evaluate them, and save individual outputs
    run_training_pipeline(df)

    # Combine prediction outputs into one final CSV
    create_final_combined_output()


if __name__ == "__main__":
    main()