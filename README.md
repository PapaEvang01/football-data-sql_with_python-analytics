# football-data-sql_with_python-analytics

## Football Data Analytics & Match Prediction Pipeline

This project explores Premier League football data using a complete analytics workflow that combines:

- SQL analytics
- Python exploratory data analysis (EDA)
- Machine Learning prediction

The goal of the project was to understand how football match statistics can be transformed from raw data into structured analytics and predictive sports models.

---

# Project Motivation

The project initially started as an SQL analytics exercise focused on exploring football match statistics and understanding team performance patterns.

Using SQL, raw football data was transformed into structured analytical summaries related to:
- team performance
- match outcomes
- goals and scoring behavior
- historical rivalries and matchups

After the analytical phase, Python was used for:
- visualization
- exploratory data analysis
- correlation analysis
- statistical pattern exploration

Finally, the project evolved into a Machine Learning pipeline where engineered football features were used to predict:
- match outcomes
- Over/Under 2.5 goals

The workflow follows this structure:

```text
Raw Football Data
        ↓
SQL Analytics
        ↓
Python EDA & Visualization
        ↓
Feature Engineering
        ↓
Machine Learning Prediction
```

---

# Dataset

### Dataset:
Premier League Matches (2010–2020)

### Source:
Kaggle

The dataset contains:
- match results
- goals scored
- possession statistics
- shots and shots on target
- fouls and cards
- passing statistics
- defensive actions
- home/away team information

Each row represents one football match.

---

# Project Structure

```text
football-data-sql_with_python-analytics/
│
├── data/
│
├── part_1_SQL/
│   ├── code/
│   ├── results/
│   └── descriptions/
│
├── part_1_python_EDA/
│   ├── code/
│   ├── results/
│   └── descriptions/
│
├── part_2_ML/
│   ├── code/
│   ├── data/
│   ├── results/
│   └── descriptions/
│
└── README.md
```

---

# Part – SQL Analytics

The SQL phase focused on transforming raw football match data into structured analytical summaries.

The analysis included:
- dataset overview
- team performance analysis
- matchup analysis
- scoring behavior
- home vs away performance
- rivalry statistics

Key SQL techniques used:
- GROUP BY
- CASE WHEN
- UNION ALL
- aggregate functions
- filtering and ranking

The final SQL outputs were exported as CSV files and later used in Python and Machine Learning.

---

# Part 1 – Python EDA & Visualization

Python was used to visualize and analyze the structured SQL outputs.

### Libraries:
- Pandas
- Matplotlib
- Seaborn

The EDA phase included:
- ranking charts
- scatter plots
- heatmaps
- matchup analysis
- correlation analysis
- attacking vs defensive comparisons

The analysis revealed patterns related to:
- team consistency
- goal production
- home advantage
- rivalry intensity
- relationships between football metrics

---

# Part 2 – Machine Learning

The final phase of the project focused on predictive football analytics.

Using engineered football features, Machine Learning models were trained to predict:
- match outcomes (W1 / Draw / W2)
- Over/Under 2.5 goals

### Implemented Models:
- Logistic Regression
- Random Forest Classifier

### Engineered Features:
- team win rates
- average goals scored
- goal difference
- comparative strength metrics
- matchup intensity
- historical rivalry statistics

---

# Machine Learning Results

## Match Outcome Prediction
### Model:
Logistic Regression

### Final Accuracy:
- 40.74%

The model performed best when predicting home wins and captured useful patterns related to:
- team strength
- goal difference
- historical performance

---

## Over/Under 2.5 Goals Prediction
### Model:
Random Forest

### Final Accuracy:
- 50.00%

The model achieved balanced performance across both classes and handled non-linear relationships between football features effectively.

---

# Final Prediction Output

The final Machine Learning pipeline generates:

```text
final_combined_predictions.csv
```

This file combines:
- predicted match outcomes
- Over/Under predictions
- probability scores
- confidence metrics

### Included Columns:
- team_1
- team_2
- W1_prob
- D_prob
- W2_prob
- predicted_match_result
- Under_prob
- Over_prob
- predicted_over_under
- confidence

The final output provides a compact probability-based summary of football match predictions.

---

# Technologies Used

- SQL
- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

# Key Takeaways

This project demonstrates how football data can be transformed from:
- raw match records

into:
- structured analytics
- visual insights
- predictive Machine Learning models

The workflow combines:
- SQL analytics
- Python data analysis
- feature engineering
- classification modeling
- probability-based prediction analysis

into a complete football analytics pipeline.

---

# Future Improvements

Possible future extensions include:
- larger football datasets
- advanced ML models (XGBoost, Neural Networks)
- expected goals (xG) features
- player-level analytics
- deep learning approaches
- time-series football analysis

---

# Author

## Evangelos Papaioannou

Electrical & Computer Engineering Graduate  
Specialization: AI & Machine Learning
