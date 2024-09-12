import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from getData import getPastData, getL15
from xgboost import XGBRegressor
from nba_api.stats.static import players
import os
import matplotlib.pyplot as plt


def getModel():
    if not os.path.isfile("/Users/sayokbose/Desktop/projects/nba/final_data.csv"):
        final_df = getPastData('2023-24')
    else:
        final_df = pd.read_csv('final_data.csv')
    final_df['TEAM_ID'] = final_df['TEAM_ID'].astype(int)
    final_df['MATCHUP_ID'] = final_df['MATCHUP_ID'].astype(int)
    final_df.to_csv('final_data.csv', index=False)

    # Split the data into features (X) and target variable (y)
    X = final_df.drop(columns=['EPTS'])  # Features
    y = final_df['EPTS'] 

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest model
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred_rf = rf_model.predict(X_test)

    # Add the predicted scores to a DataFrame
    predictions_df = pd.DataFrame({'Actual_EPTS': y_test, 'Predicted_EPTS_RF': y_pred_rf})

    # Initialize and train the Gradient Boosting model
    gb_model = GradientBoostingRegressor(random_state=42)
    gb_model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred_gb = gb_model.predict(X_test)

    feature_importances_rf = rf_model.feature_importances_

    feature_importances_gb = gb_model.feature_importances_
    # Random Forest Feature Importance
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(X.columns, feature_importances_rf)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Random Forest Feature Importance')

    # Gradient Boost Feature Importance
    plt.subplot(1, 2, 2)
    plt.barh(X.columns, feature_importances_gb)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Gradient Boost Feature Importance')

    plt.tight_layout()
    plt.show()










    mse_rf = mean_squared_error(y_test, y_pred_rf)
    mse_gb = mean_squared_error(y_test, y_pred_gb)

    print("Random Forest Mean Squared Error (MSE):", mse_rf)
    print("Gradient Boosting Mean Squared Error (MSE):", mse_gb)
    return rf_model, gb_model


def get_player_id(player_name):
    
    matching_players = players.find_players_by_full_name(player_name)
    if len(matching_players) == 0:
        return None
    elif len(matching_players) > 1:
        print("Multiple players found with the same name. Please specify.")
        return None
    else:
        return matching_players[0]['id']


def predict(players,rf_model, gb_model):
    player_ids = [get_player_id(player_name) for player_name in players_]
    print(player_ids)
    season = '2023-24'

    # Add the predicted scores to the DataFrame
    new_data = getL15(season, player_ids)
    y_pred_rf_new = rf_model.predict(new_data)
    y_pred_gb_new = gb_model.predict(new_data)

    player_names = []
    #get player names
    for player_id in new_data['PLAYER_ID']:
        player_info = players.find_player_by_id(player_id)
        player_name = player_info['full_name']
        player_names.append(player_name)
    # Add the predicted scores to a DataFrame
    predictions_df = pd.DataFrame({'PLAYER_NAME': player_names,'Predicted_EPTS_RF': y_pred_rf_new, 'Predicted_EPTS_GB': y_pred_gb_new})

    # Print the DataFrame with predicted scores
    print(predictions_df)

    # Optionally, save the predictions to a CSV file
    predictions_df.to_csv('predictions_data.csv', index=False)


    
players_ = ['Alperen Sengun', 'Jalen Green', 'Domantas Sabonis', 'Dillon Brooks','Jabari Smith','Fred VanVleet', 'Cam Whitmore', "De'Aaron Fox","Giannis Antetokounmpo", "Malik Beasley","Damian Lillard","Brook Lopez","James Harden","Kawhi Leonard"]
models = getModel()
predict(players, models[0], models[1])







