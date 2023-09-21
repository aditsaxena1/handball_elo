import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import requests
import json
from pandas import json_normalize

# Initial Variables that affect ELO
initial_rating = 1500
K = 16 # factor at which scores are updated
home_advantage = 32 #should be in number of elo points

def sigmoid_strength_of_victory(goal_dif, equation = 1, elo_diff= 0):
    """
    A team that wins by more should be rewarded for their victory more
    but we don't want to make extremely large wins too overvalued
    so a sigmoid function was chosen to represent this
    goal_dif: goal difference for the match for the winning team
    equation: 
    """
    if equation == 1: 
        sov = (6.2 / (2 + math.exp(-(1 / 5) * (goal_dif - 8))))
    elif equation == 2:
        # 538 Model for Phase 2
        sov = np.log(goal_dif + 1) * (2.2 / (elo_diff + 2.2))
    return sov

def calculate_expected(team1, team2):
    """
    Function to calculate expected outcome with home team advantage

    row: a row from pandas DataFrame, where row['home_elo_before_match'] is home team rating (elo rating)
        and row['away_elo_before_match'] is away team rating (elo rating)

    home_advantage: numeric value representing the home team's advantage

    """
    team1_expected = 1 / (1 + math.pow(10, (team2 - (team1 + home_advantage)) / 400))
    return team1_expected

def update_ratings(home_team_rating, away_team_rating, goal_difference):
    """
    Function to update ratings based on match outcome and goal difference
    """
    outcome = 0.5 if goal_difference == 0 else 1 if goal_difference > 0 else 0
    #expected = win_prob
    expected = calculate_expected(home_team_rating, away_team_rating)

    strength_of_victory = sigmoid_strength_of_victory(goal_difference)

    return home_team_rating + (K * (strength_of_victory) * (outcome - expected))

def update_away_ratings(home_team_rating, away_team_rating, goal_difference):
    outcome = 0.5 if goal_difference == 0 else 1 if abs(goal_difference) > 0 else 0
    expected = 1 / (1 + math.pow(10, ((home_team_rating + home_advantage) - away_team_rating) / 400))
    strength_of_victory = sigmoid_strength_of_victory(abs(goal_difference))
    return (away_team_rating + K * (strength_of_victory) * (outcome - expected))

def clean_df(df):
    df = df[["date", "week", "teams.home.name", "scores.home", "teams.away.name", "scores.away", "league.season"]]
    df['difference'] = df['scores.home'] - df['scores.away']
    df = df[df['week'] != "Final"]
    df['week'] = df['week'].astype(int)
    return df

results = []
percentages = []
max_K = 1600
while K <= max_K:
    # Open and load the JSON file
    with open('data/game_results.json') as f:
        data = json.load(f)

# Convert the JSON data to a DataFrame
    df = pd.json_normalize(data)

    df = clean_df(df)

    # Create a dictionary to store the latest ELO rating for each team
    elo_dict = {}
    end_of_season_elo_dict = {}

    # Initialize a variable to keep track of the current season
    current_season = df['league.season'].iloc[0]

    # The regression factor and the initial Elo rating for new teams
    regression_factor = 0.67
    initial_elo = 1500

    # Get the set of all teams
    teams = set(df['teams.home.name']).union(set(df['teams.away.name']))

    # Create empty lists to store the new ELO ratings and the ratings before the match
    home_elo_ratings = []
    away_elo_ratings = []
    home_elo_before_match = []
    away_elo_before_match = []

    # Go through each row in the dataframe
    for idx, row in df.iterrows():

        home_team = row['teams.home.name']
        away_team = row['teams.away.name']
        score_diff = row['difference']
        
        # Check if the home team is in the Elo ratings dictionary
        if home_team not in elo_dict:
            elo_dict[home_team] = 1500

        # Check if the away team is in the Elo ratings dictionary
        if away_team not in elo_dict:
            elo_dict[away_team] = 1500

        # Check if the season has changed
        if row['league.season'] != current_season:
            # If the season has changed, regress the Elo ratings towards the mean
            end_of_season_elo_dict[current_season] = sum(elo_dict.values())
            for team in elo_dict:
                elo_dict[team] = regression_factor * (elo_dict[team] - initial_elo) + initial_elo

            # Update the current season
            current_season = row['league.season']

        # Store the ratings before the match
        home_elo_before_match.append(elo_dict[home_team])
        away_elo_before_match.append(elo_dict[away_team])
        
        if score_diff >= 0: 
            new_home_elo = update_ratings(elo_dict[home_team], elo_dict[away_team], score_diff)
            new_away_elo = elo_dict[away_team] - (new_home_elo - elo_dict[home_team])
        
        else: 
            new_away_elo = update_away_ratings(elo_dict[home_team], elo_dict[away_team], score_diff)
            new_home_elo = elo_dict[home_team] - (new_away_elo - elo_dict[away_team])

        home_elo_ratings.append(new_home_elo)
        away_elo_ratings.append(new_away_elo)

            # Update the ELO ratings in our dictionary
        elo_dict[home_team] = new_home_elo
        elo_dict[away_team] = new_away_elo

    # Add the new ELO ratings to the dataframe
    df['home_elo_rating'] = home_elo_ratings
    df['away_elo_rating'] = away_elo_ratings

    # Add the ELO ratings before the match to the dataframe
    df['home_elo_before_match'] = home_elo_before_match
    df['away_elo_before_match'] = away_elo_before_match

    # Calculate the expected outcome (win probability) for the home team and the away team
    df['home_win_probability'] = 1 / (1 + 10**(((df['away_elo_before_match'] - (df['home_elo_before_match'] + home_advantage)) / 400))) * 100
    df['away_win_probability'] = 100 - df['home_win_probability']

    df['home_team_won'] = np.where(df['scores.home'] > df['scores.away'], 'Won',
                                np.where(df['scores.home'] < df['scores.away'], 'Lost', 'Tied'))

    # Create a column to check if the home_win_probability was correct
    # If home_win_probability > 50, the model predicts home team to win
    # If home_win_probability < 50, the model predicts home team to lose
    # If it's a tie, the prediction is considered correct only if home_win_probability is exactly 50%
    conditions = (
        ((df['home_win_probability'] > 50) & (df['home_team_won'] == 'Won')) |
        ((df['home_win_probability'] < 50) & (df['home_team_won'] == 'Lost')) |
        ((df['home_win_probability'] == 50) & (df['home_team_won'] == 'Tied'))
    )
    df['prediction_correct'] = np.where(conditions, 'Yes', 'No')

        # Create a column to indicate the bin for each game's home win probability
    df['probability_bin'] = pd.cut(df['home_win_probability'], bins=np.arange(0, 110, 10))

# Group by the probability bin and calculate the percentage of correct predictions in each bin
    accuracy_by_bin = df.groupby('probability_bin')['prediction_correct'].apply(lambda x: (x == 'Yes').mean() * 100)

# Group by the probability bin, calculate the percentage of correct predictions 
# in each bin and also count the number of predictions
    grouped = df.groupby('probability_bin').agg({'prediction_correct': lambda x: (x == 'Yes').mean() * 100,
                                             'home_win_probability': 'count'})

# Rename the columns for clarity
    grouped.columns = ['Accuracy (%)', 'Sample Size']
    #print(K)
    initial_rating = initial_rating + 1
    results.append(grouped)
    #print(results[-1])
    yes_count = 0
    no_count = 0

    for entry in df.prediction_correct:
        if entry.lower() == "yes":
            yes_count += 1
        elif entry.lower() == "no":
            no_count += 1
    #print(yes_count/ (yes_count + no_count) * 100)
    #print(K-1, yes_count/ (yes_count + no_count) * 100)
    percentages.append(yes_count/ (yes_count + no_count) * 100)
print(len(percentages))
plt.scatter(np.linspace(800, len(percentages), len(percentages)), percentages)
plt.xlabel("Scale Factor of points (K)")
plt.ylabel("Model Accuracy")
plt.title('Tuning Scaling Factor')
plt.show()