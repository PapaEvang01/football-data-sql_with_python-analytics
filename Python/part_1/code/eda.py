# ============================================================
# Football Data Analytics - Exploratory Data Analysis (EDA)
# ============================================================
#
# Description:
# This script performs Exploratory Data Analysis (EDA)
# using the analytical CSV outputs generated from the
# SQL section of the project.
#
# The goal is to transform structured football analytics
# data into visual insights using Python libraries such
# as pandas, matplotlib, and seaborn.
#
# The analysis focuses on:
# - General dataset statistics
# - Team performance analysis
# - Matchup and rivalry analysis
#
# Generated visualizations are automatically saved into
# categorized result folders for easier organization
# and interpretation.
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# FILE PATHS
# ============================================================
OVERALL_PATH = "data/04_01.csv"
TEAM_PATH = "data/04_02.csv"
MATCHUP_PATH = "data/04_03_matchup_summary.csv"


# ============================================================
# OUTPUT FOLDERS
# ============================================================
GENERAL_DIR = "results/figures/01_general"
TEAM_DIR = "results/figures/02_team_perf"
MATCHUP_DIR = "results/figures/03_matchup"

# ============================================================
# GENERAL FUNCTIONS
# ============================================================
# Utility functions used throughout the EDA workflow
# for folder creation, CSV loading, dataset inspection,
# and chart formatting.

def create_output_folders():
    """Create structured output folders."""
    os.makedirs(GENERAL_DIR, exist_ok=True)
    os.makedirs(TEAM_DIR, exist_ok=True)
    os.makedirs(MATCHUP_DIR, exist_ok=True)


def load_csv(file_path, name):
    """Load a CSV file and return a DataFrame."""
    try:
        df = pd.read_csv(file_path)

        print(f"[OK] {name} loaded successfully.")

        return df

    except Exception as e:
        print(f"[ERROR] Failed to load {name}: {e}")
        return None


def print_basic_info(df, name):
    """Display basic dataset information."""
    if df is None:
        print(f"[WARNING] {name} is None.")
        return

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Shape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nFirst 5 rows:")
    print(df.head())


def format_label(value):
    """Format chart labels for cleaner visualization."""
    return f"{int(value)}" if float(value).is_integer() else f"{value:.2f}"

# ============================================================
# 04_01.csv - OVERALL DATASET ANALYSIS
# ============================================================
# This section visualizes general dataset statistics
# generated from the SQL analytics workflow.
#
# The goal is to create a compact dashboard-style
# overview of the dataset using key numerical metrics.

def analyze_overall(overall_df):
    """Create a dashboard-style chart for overall dataset metrics."""

    if overall_df is None:
        print("[WARNING] Overall Summary is missing.")
        return

    print("\n" + "=" * 60)
    print("OVERALL DATASET ANALYSIS")
    print("=" * 60)

    # Create a safe copy of the dataset
    df = overall_df.copy()

    # Convert values to numeric format when possible
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Keep only valid numerical rows for visualization
    numeric_df = df.dropna().copy()

    print("\nNumeric Metrics Used for Plot:")
    print(numeric_df)

    # Create horizontal bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.barh(numeric_df["metric"], numeric_df["value"])

    # Add value labels next to each bar
    for bar in bars:
        width = bar.get_width()

        plt.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            format_label(width),
            va="center"
        )

    # Chart formatting
    plt.title("Dataset Overview")
    plt.xlabel("Value")
    plt.ylabel("Metric")
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(GENERAL_DIR, "overall_dashboard.png"))

    # Display figure
    plt.show()

# ============================================================
# 04_02.csv - TEAM PERFORMANCE ANALYSIS
# ============================================================
# This section analyzes team-level performance using the
# final team summary table generated from SQL.
#
# The visualizations focus on:
# - strongest teams by wins and win rate
# - attacking and defensive profiles
# - home vs away performance
# - relationships between key performance metrics


def plot_top_teams(team_df, column, title, filename):
    """Create a horizontal bar chart for the top 10 teams by a selected metric."""

    # Select the top 10 teams based on the chosen metric
    top = team_df.sort_values(by=column, ascending=False).head(10)

    # Create horizontal bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.barh(top["team"], top[column])

    # Display the strongest team at the top of the chart
    plt.gca().invert_yaxis()

    # Add value labels next to each bar
    for bar in bars:
        width = bar.get_width()

        plt.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            format_label(width),
            va="center"
        )

    # Chart formatting
    plt.title(title)
    plt.xlabel(column)
    plt.tight_layout()

    # Save and display figure
    plt.savefig(os.path.join(TEAM_DIR, filename))
    plt.show()


def plot_home_vs_away_win_rate(team_df):
    """Compare home and away win rates for the top teams by overall win rate."""

    # Select the top 10 teams based on overall win rate
    top_teams = team_df.sort_values(by="win_rate_pct", ascending=False).head(10)

    plt.figure(figsize=(10, 6))

    # X-axis positions for grouped bars
    x = range(len(top_teams))

    # Plot home win rate
    plt.bar(
        x,
        top_teams["home_win_rate_pct"],
        width=0.4,
        label="Home Win %",
        align="center"
    )

    # Plot away win rate next to home win rate
    plt.bar(
        [i + 0.4 for i in x],
        top_teams["away_win_rate_pct"],
        width=0.4,
        label="Away Win %"
    )

    # Add team names as x-axis labels
    plt.xticks(
        [i + 0.2 for i in x],
        top_teams["team"],
        rotation=45,
        ha="right"
    )

    # Chart formatting
    plt.title("Home vs Away Win Rate (Top Teams)")
    plt.ylabel("Win Rate (%)")
    plt.legend()
    plt.tight_layout()

    # Save and display figure
    plt.savefig(os.path.join(TEAM_DIR, "home_vs_away_win_rate.png"))
    plt.show()


def plot_attacking_vs_defensive(team_df):
    """Create a scatter plot comparing attacking and defensive team profiles."""

    plt.figure(figsize=(10, 6))

    # Each point represents one team
    plt.scatter(
        team_df["avg_goals_scored"],
        team_df["avg_goals_conceded"]
    )

    # Add team labels next to each point
    for _, row in team_df.iterrows():
        plt.text(
            row["avg_goals_scored"] + 0.01,
            row["avg_goals_conceded"] + 0.01,
            row["team"],
            fontsize=8
        )

    # Chart formatting
    plt.title("Attacking vs Defensive Team Profile")
    plt.xlabel("Average Goals Scored")
    plt.ylabel("Average Goals Conceded")
    plt.tight_layout()

    # Save and display figure
    plt.savefig(os.path.join(TEAM_DIR, "teams_attack_vs_defense_scatter.png"))
    plt.show()


def plot_winrate_vs_goaldiff(team_df):
    """Create a scatter plot comparing win rate and average goal difference."""

    plt.figure(figsize=(10, 6))

    # Each point represents one team
    plt.scatter(
        team_df["avg_goal_difference"],
        team_df["win_rate_pct"]
    )

    # Add team labels next to each point
    for _, row in team_df.iterrows():
        plt.text(
            row["avg_goal_difference"] + 0.02,
            row["win_rate_pct"] + 0.2,
            row["team"],
            fontsize=8
        )

    # Chart formatting
    plt.title("Win Rate vs Goal Difference")
    plt.xlabel("Average Goal Difference")
    plt.ylabel("Win Rate (%)")
    plt.tight_layout()

    # Save and display figure
    plt.savefig(os.path.join(TEAM_DIR, "winrate_vs_goaldiff.png"))
    plt.show()


def plot_team_correlation_heatmap(team_df):
    """Create a correlation heatmap for key team performance metrics."""

    # Select numerical performance metrics
    columns = [
        "total_wins",
        "win_rate_pct",
        "avg_goals_scored",
        "avg_goals_conceded",
        "avg_goal_difference"
    ]

    # Compute correlation matrix
    corr_df = team_df[columns].corr()

    # Create heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f")

    # Chart formatting
    plt.title("Correlation Heatmap of Team Performance Metrics")
    plt.tight_layout()

    # Save and display figure
    plt.savefig(os.path.join(TEAM_DIR, "team_correlation_heatmap.png"))
    plt.show()


def plot_home_away_heatmap(team_df):
    """Create a correlation heatmap for home and away performance metrics."""

    # Select home and away result-rate metrics
    columns = [
        "home_win_rate_pct",
        "home_draw_rate_pct",
        "home_loss_rate_pct",
        "away_win_rate_pct",
        "away_draw_rate_pct",
        "away_loss_rate_pct"
    ]

    # Compute correlation matrix
    corr_df = team_df[columns].corr()

    # Create heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f")

    # Chart formatting
    plt.title("Home vs Away Performance Correlation")
    plt.tight_layout()

    # Save and display figure
    plt.savefig(os.path.join(TEAM_DIR, "home_away_correlation.png"))
    plt.show()


def analyze_teams(team_df):
    """Run all team performance visualizations."""

    if team_df is None:
        print("[WARNING] Team Summary is missing.")
        return

    print("\n" + "=" * 60)
    print("TEAM ANALYSIS")
    print("=" * 60)

    # Ranking charts
    plot_top_teams(
        team_df,
        "total_wins",
        "Top 10 Teams by Total Wins",
        "teams_top_wins.png"
    )

    plot_top_teams(
        team_df,
        "win_rate_pct",
        "Top 10 Teams by Win Rate (%)",
        "teams_win_rate.png"
    )

    plot_top_teams(
        team_df,
        "avg_goals_scored",
        "Top 10 Teams by Avg Goals Scored",
        "teams_avg_goals.png"
    )

    plot_top_teams(
        team_df,
        "avg_goal_difference",
        "Top 10 Teams by Goal Difference",
        "teams_goal_diff.png"
    )

    # Comparison and relationship charts
    plot_home_vs_away_win_rate(team_df)
    plot_attacking_vs_defensive(team_df)
    plot_winrate_vs_goaldiff(team_df)

    # Correlation heatmaps
    plot_team_correlation_heatmap(team_df)
    plot_home_away_heatmap(team_df)

# ============================================================
# 04_03_matchup_summary.csv - BIG 5 MATCHUP ANALYSIS
# ============================================================
# This section analyzes head-to-head matchups between
# selected Big 5 Premier League clubs.
#
# The analysis focuses on:
# - average goals per matchup
# - match intensity based on fouls and cards
# - overall matchup performance metrics
# - win-rate dominance between teams


def analyze_matchups(matchup_df):
    """Analyze matchups only between selected Big 5 clubs."""

    if matchup_df is None:
        print("[WARNING] Matchup Summary is missing.")
        return

    print("\n" + "=" * 60)
    print("BIG 5 MATCHUP ANALYSIS")
    print("=" * 60)

    # Define the selected Big 5 clubs used for matchup analysis
    big_teams = [
        "Arsenal",
        "Manchester City",
        "Manchester United",
        "Chelsea",
        "Liverpool"
    ]

    # Keep only matchups where both teams belong to the Big 5 list
    df_big = matchup_df[
        matchup_df["team_1"].isin(big_teams) &
        matchup_df["team_2"].isin(big_teams)
    ].copy()

    # Create a readable matchup label for charts
    df_big["matchup"] = df_big["team_1"] + " vs " + df_big["team_2"]

    print(f"\nTotal Big 5 Matchups Analyzed: {len(df_big)}")
    print(df_big[["matchup", "total_matches", "avg_goals", "avg_fouls", "avg_shots", "avg_corners"]])

    # Generate all Big 5 matchup visualizations
    plot_big5_goals(df_big)
    plot_big5_intensity(df_big)
    plot_big5_heatmap(df_big)
    plot_big5_dominance(df_big)


def plot_big5_goals(df_big):
    """Plot average goals for Big 5 matchups."""

    # Sort matchups by average goals scored
    top = df_big.sort_values(by="avg_goals", ascending=False)

    # Create horizontal bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.barh(top["matchup"], top["avg_goals"])

    # Display the highest-scoring matchup at the top
    plt.gca().invert_yaxis()

    # Add value labels next to each bar
    for bar in bars:
        width = bar.get_width()

        plt.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            format_label(width),
            va="center"
        )

    # Chart formatting
    plt.title("Big 5 Matchups - Average Goals")
    plt.xlabel("Average Goals")
    plt.tight_layout()

    # Save and display figure
    plt.savefig(os.path.join(MATCHUP_DIR, "big5_avg_goals.png"))
    plt.show()


def plot_big5_intensity(df_big):
    """Plot intensity score for Big 5 matchups."""

    # Work on a copy to avoid modifying the original DataFrame
    df_big = df_big.copy()

    # Create a simple custom intensity score.
    # Red cards are weighted more heavily because they have
    # a stronger impact on match dynamics.
    df_big["intensity_score"] = (
        df_big["avg_fouls"]
        + df_big["avg_yellow_cards"]
        + df_big["avg_red_cards"] * 2
    )

    # Sort matchups by intensity score
    top = df_big.sort_values(by="intensity_score", ascending=False)

    # Create horizontal bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.barh(top["matchup"], top["intensity_score"])

    # Display the most intense matchup at the top
    plt.gca().invert_yaxis()

    # Add value labels next to each bar
    for bar in bars:
        width = bar.get_width()

        plt.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            format_label(width),
            va="center"
        )

    # Chart formatting
    plt.title("Big 5 Matchups - Intensity Score")
    plt.xlabel("Intensity Score")
    plt.tight_layout()

    # Save and display figure
    plt.savefig(os.path.join(MATCHUP_DIR, "big5_intensity_score.png"))
    plt.show()


def plot_big5_heatmap(df_big):
    """Create a heatmap for Big 5 matchup metrics."""

    # Select matchup metrics for comparison
    columns = [
        "avg_goals",
        "avg_fouls",
        "avg_shots",
        "avg_corners"
    ]

    # Use matchup names as heatmap rows
    heatmap_data = df_big.set_index("matchup")[columns]

    # Create heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        heatmap_data,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    # Chart formatting
    plt.title("Big 5 Matchup Performance Overview")
    plt.tight_layout()

    # Save and display figure
    plt.savefig(os.path.join(MATCHUP_DIR, "big5_matchup_heatmap.png"))
    plt.show()


def plot_big5_dominance(df_big):
    """Show win-rate dominance differences in Big 5 matchups."""

    # Work on a copy to preserve the original DataFrame
    df = df_big.copy()

    # Calculate absolute difference between team win rates
    df["dominance"] = abs(
        df["team_1_win_rate_pct"] - df["team_2_win_rate_pct"]
    )

    # Create horizontal bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.barh(df["matchup"], df["dominance"])

    # Display the largest dominance difference at the top
    plt.gca().invert_yaxis()

    # Add value labels next to each bar
    for bar in bars:
        width = bar.get_width()

        plt.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            va="center"
        )

    # Chart formatting
    plt.title("Big 5 Matchups - Win Dominance")
    plt.xlabel("Win Rate Difference (%)")
    plt.tight_layout()

    # Save and display figure
    plt.savefig(os.path.join(MATCHUP_DIR, "big5_dominance.png"))
    plt.show()
    
# ============================================================
# MAIN
# ============================================================
# Main execution pipeline.
#
# The script follows this order:
# 1. Create output folders
# 2. Load SQL-generated CSV files
# 3. Print basic dataset information
# 4. Generate all EDA visualizations


def main():
    """Run the complete Python EDA workflow."""

    # Create folders for saving figures
    create_output_folders()

    # Load analytical CSV outputs generated from SQL
    overall_df = load_csv(OVERALL_PATH, "Overall Summary")
    team_df = load_csv(TEAM_PATH, "Team Summary")
    matchup_df = load_csv(MATCHUP_PATH, "Matchup Summary")

    # Inspect loaded datasets
    print_basic_info(overall_df, "Overall Summary")
    print_basic_info(team_df, "Team Summary")
    print_basic_info(matchup_df, "Matchup Summary")

    # Run analysis and generate visualizations
    analyze_overall(overall_df)
    analyze_teams(team_df)
    analyze_matchups(matchup_df)


if __name__ == "__main__":
    main()
