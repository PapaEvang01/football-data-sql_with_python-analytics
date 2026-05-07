import pandas as pd
from pathlib import Path


# ============================================================
# Football Match Prediction - ML Dataset Builder
# ============================================================
#
# Description:
# This script builds the Machine Learning dataset used
# for football match outcome prediction.
#
# The dataset is created using the structured analytical
# outputs generated from the SQL workflow, focusing on
# matchups between selected Big 5 Premier League clubs.
#
# The goal is to:
# - prepare clean ML-ready data
# - engineer predictive football features
# - generate target variables for classification tasks
#
# The final dataset is later used for:
# - match outcome prediction
# - Over/Under goals prediction
# - football analytics modeling
# ============================================================


# ============================================================
# SELECTED BIG 5 PREMIER LEAGUE CLUBS
# ============================================================
# These teams are used to filter the dataset and create
# focused high-level football matchup analysis.

BIG_TEAMS = [
    "Arsenal",
    "Manchester City",
    "Manchester United",
    "Chelsea",
    "Liverpool"
]

# ============================================================
# FILE PATHS
# ============================================================
# Input datasets generated from the SQL analytics pipeline
# and the original football match dataset.

DATA_DIR = Path("data")
OUTPUT_DIR = Path("data/ml_ready")

MATCHES_PATH = DATA_DIR / "football_matches.csv"
TEAM_SUMMARY_PATH = DATA_DIR / "04_02.csv"
MATCHUP_SUMMARY_PATH = DATA_DIR / "04_03_matchup_summary.csv"

# Final Machine Learning dataset output
OUTPUT_PATH = OUTPUT_DIR / "ml_1_big5_match_outcome_dataset.csv"


# ============================================================
# LOAD DATASETS
# ============================================================
# Load all required datasets used for feature engineering
# and Machine Learning dataset construction.

def load_datasets():

    matches_df = pd.read_csv(MATCHES_PATH)
    team_df = pd.read_csv(TEAM_SUMMARY_PATH)
    matchup_df = pd.read_csv(MATCHUP_SUMMARY_PATH)

    return matches_df, team_df, matchup_df


# ============================================================
# FILTER BIG 5 MATCHES
# ============================================================
# Keep only matches where both teams belong to the
# selected Big 5 Premier League clubs.

def filter_big5_matches(matches_df):

    big5_matches = matches_df[
        (matches_df["home_team"].isin(BIG_TEAMS)) &
        (matches_df["away_team"].isin(BIG_TEAMS))
    ].copy()

    return big5_matches
# ============================================================
# CREATE TARGET COLUMNS
# ============================================================
# Create the prediction targets used by the Machine Learning
# models:
# - target_result: match outcome classification
# - target_over_2_5: Over/Under 2.5 goals classification

def create_target(df):

    # Define match outcome from the home team perspective
    def get_result(row):
        if row["goal_home_ft"] > row["goal_away_ft"]:
            return "W1"
        elif row["goal_home_ft"] < row["goal_away_ft"]:
            return "W2"
        else:
            return "D"

    # Match result target:
    # W1 -> team_1/home team wins
    # D  -> draw
    # W2 -> team_2/away team wins
    df["target_result"] = df.apply(get_result, axis=1)

    # Over/Under 2.5 goals target
    df["target_over_2_5"] = df.apply(
        lambda row: "Over" if (row["goal_home_ft"] + row["goal_away_ft"]) > 2.5 else "Under",
        axis=1
    )

    return df


# ============================================================
# PREPARE BASE ML DATASET
# ============================================================
# Select the basic match information and target columns
# required for the Machine Learning dataset.

def prepare_base_dataset(big5_matches):

    ml_df = big5_matches[[
        "season",
        "date",
        "home_team",
        "away_team",
        "target_result",
        "target_over_2_5"
    ]].copy()

    # Rename teams into a neutral prediction format
    ml_df = ml_df.rename(columns={
        "home_team": "team_1",
        "away_team": "team_2"
    })

    return ml_df


# ============================================================
# MERGE TEAM PERFORMANCE FEATURES
# ============================================================
# Add team-level statistics from the SQL team summary table.
# The same feature set is merged twice:
# - once for team_1
# - once for team_2

def merge_team_features(ml_df, team_df):

    # Select team-level features used as model inputs
    selected_features = [
        "team",
        "win_rate_pct",
        "avg_goal_difference",
        "avg_goals_scored"
    ]

    team_features = team_df[selected_features].copy()

    # Merge performance features for team_1
    ml_df = ml_df.merge(
        team_features,
        left_on="team_1",
        right_on="team",
        how="left"
    )

    ml_df = ml_df.rename(columns={
        "win_rate_pct": "team_1_win_rate_pct",
        "avg_goal_difference": "team_1_avg_goal_difference",
        "avg_goals_scored": "team_1_avg_goals_scored"
    })

    ml_df = ml_df.drop(columns=["team"])

    # Merge performance features for team_2
    ml_df = ml_df.merge(
        team_features,
        left_on="team_2",
        right_on="team",
        how="left"
    )

    ml_df = ml_df.rename(columns={
        "win_rate_pct": "team_2_win_rate_pct",
        "avg_goal_difference": "team_2_avg_goal_difference",
        "avg_goals_scored": "team_2_avg_goals_scored"
    })

    ml_df = ml_df.drop(columns=["team"])

    return ml_df

# ============================================================
# CREATE DIFFERENCE FEATURES
# ============================================================
# Create comparative features between team_1 and team_2.
# These features help the model understand relative strength
# instead of only using each team's individual statistics.

def create_difference_features(ml_df):

    ml_df["diff_win_rate"] = (
        ml_df["team_1_win_rate_pct"] - ml_df["team_2_win_rate_pct"]
    ).round(2)

    ml_df["diff_goal_diff"] = (
        ml_df["team_1_avg_goal_difference"] - ml_df["team_2_avg_goal_difference"]
    ).round(2)

    ml_df["diff_goals_scored"] = (
        ml_df["team_1_avg_goals_scored"] - ml_df["team_2_avg_goals_scored"]
    ).round(2)

    return ml_df


# ============================================================
# MERGE MATCHUP FEATURES
# ============================================================
# Add historical matchup-level features from the SQL matchup
# summary table.
#
# A normalized matchup key is created so that:
# Arsenal vs Liverpool
# and
# Liverpool vs Arsenal
# are treated as the same matchup.

def merge_matchup_features(ml_df, matchup_df):

    matchup_features = matchup_df.copy()

    # Create normalized matchup key for the ML dataset
    ml_df["matchup_team_1"] = ml_df[["team_1", "team_2"]].min(axis=1)
    ml_df["matchup_team_2"] = ml_df[["team_1", "team_2"]].max(axis=1)

    # Create normalized matchup key for the matchup summary dataset
    matchup_features["matchup_team_1"] = matchup_features[["team_1", "team_2"]].min(axis=1)
    matchup_features["matchup_team_2"] = matchup_features[["team_1", "team_2"]].max(axis=1)

    # Create a custom intensity score based on fouls and cards
    matchup_features["matchup_intensity_score"] = (
        matchup_features["avg_fouls"] +
        matchup_features["avg_yellow_cards"] +
        2 * matchup_features["avg_red_cards"]
    ).round(2)

    # Create a dominance score based on average goal difference
    matchup_features["matchup_dominance"] = (
        matchup_features["avg_goal_difference"].abs()
    ).round(2)

    # Keep only matchup features required for the ML dataset
    matchup_features = matchup_features[[
        "matchup_team_1",
        "matchup_team_2",
        "avg_goals",
        "matchup_intensity_score",
        "matchup_dominance"
    ]].copy()

    matchup_features = matchup_features.rename(columns={
        "avg_goals": "matchup_avg_goals"
    })

    # Merge matchup-level features into the ML dataset
    ml_df = ml_df.merge(
        matchup_features,
        on=["matchup_team_1", "matchup_team_2"],
        how="left"
    )

    # Remove temporary normalized matchup columns
    ml_df = ml_df.drop(columns=["matchup_team_1", "matchup_team_2"])

    return ml_df


# ============================================================
# MAIN
# ============================================================
# Main execution pipeline:
# 1. Create output directory
# 2. Load source datasets
# 3. Filter Big 5 matches
# 4. Create target variables
# 5. Merge team-level features
# 6. Create comparative features
# 7. Merge matchup-level features
# 8. Save final ML-ready dataset

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load original match data and SQL-generated summary datasets
    matches_df, team_df, matchup_df = load_datasets()

    # Keep only Big 5 head-to-head matches
    big5_matches = filter_big5_matches(matches_df)

    # Create classification target columns
    big5_matches = create_target(big5_matches)

    # Build base ML dataset
    ml_df = prepare_base_dataset(big5_matches)

    # Add team-level and matchup-level predictive features
    ml_df = merge_team_features(ml_df, team_df)
    ml_df = create_difference_features(ml_df)
    ml_df = merge_matchup_features(ml_df, matchup_df)

    # Save final ML-ready dataset
    ml_df.to_csv(OUTPUT_PATH, index=False)

    print("ML dataset created successfully.")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Rows: {len(ml_df)}")
    print(ml_df.head())


if __name__ == "__main__":
    main()