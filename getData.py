import time
from nba_api.stats.endpoints import PlayerGameLogs, TeamEstimatedMetrics, BoxScoreAdvancedV2, Scoreboard
import pandas as pd
from datetime import datetime

def compare_dates(date1, date2):
    # Parse date1 and date2 into datetime objects
    datetime1 = datetime.strptime(date1, '%Y-%m-%d')
    datetime2 = datetime.strptime(date2, '%Y-%m-%d')

    # Compare the dates
    if datetime1 < datetime2:
        return -1
    elif datetime1 == datetime2:
        return 0
    else:
        return 1
    
def addAdvanced():
    advanced_stats_cache = {}
    def get_advanced_stats(game_id):
        try:
            # Check if stats for this game_id are already cached
            if game_id in advanced_stats_cache.keys():
                print('in here')
                return advanced_stats_cache[game_id]

            # Retrieve data
            boxscore = BoxScoreAdvancedV2(
                end_period=1,
                end_range=0,
                game_id=game_id,
                range_type=0,
                start_period=1,
                start_range=0
            )
            print('wait')
            time.sleep(.500)
            print('done wait')
            data = boxscore.get_dict()

            # Extract player and team stats
            player_stats = data['resultSets'][0]['rowSet']
            team_stats = data['resultSets'][1]['rowSet']

            # Extract desired stats
            player_usg_pct = {player[5]: player[24] for player in player_stats}
            team_def_rating = {team[3]: team[8] for team in team_stats}
            team_pace = {team[3]: team[24] for team in team_stats}

            # Store stats in cache
            advanced_stats_cache[game_id] = {
                'PLAYER_USG_PCT': player_usg_pct,
                'TEAM_DEF_RATING': team_def_rating,
                'TEAM_PACE': team_pace
            }

            return advanced_stats_cache[game_id]
        
        except Exception as e:
            print("Error:", e)
            return None
        


    organized_df = pd.read_csv('organized_data.csv')

    


    # Iterate over the rows of organized_df
    for index, row in organized_df.iterrows():
        print('unique games accessed: ', len(advanced_stats_cache.keys()))
        print(index)
        game_id = row['GAME_ID']
        game_id = str(game_id).zfill(10)
        player_name = row['PLAYER_NAME']
        opp = row["MATCHUP"]
        # Get advanced stats for the game and player
        boxScore = get_advanced_stats(game_id)
        usg = boxScore['PLAYER_USG_PCT'][player_name]
        derat = boxScore['TEAM_DEF_RATING'][opp]
        pace = boxScore['TEAM_PACE'][opp]

        organized_df.at[index, 'PLAYER_USG_PCT'] = usg
        organized_df.at[index, 'TEAM_DEF_RATING'] = derat
        organized_df.at[index, 'TEAM_PACE'] = pace

    # Display the updated DataFrame

    organized_df.to_csv('advanced_data.csv', index=False)

    return organized_df





def getNBADF(season, season_type = 0, season_segment = 0, backup = False):
    # Define the parameters for retrieving player game logs
    if(season_segment != 0):
        print("seaon inside is ", season)
        player_game_logs = PlayerGameLogs(season_nullable=season, season_segment_nullable = 'Post All-Star')
    else:
        # Retrieve player game logs
        player_game_logs = PlayerGameLogs(season_nullable=season, season_type_nullable=season_type)
    df_player_game_logs = player_game_logs.get_data_frames()[0]

    # Define the list of columns to keep
    columns_to_keep = ['SEASON_YEAR', 'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION',
                    'GAME_ID', 'MATCHUP', 'WL', 'MIN', 'FGM', 'FGA', 'FG_PCT',
                    'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
                    'OREB', 'DREB', 'AST', 'TOV', 'PF', 'PTS', 'PLUS_MINUS',
                    'NBA_FANTASY_PTS',"MIN_RANK","GAME_DATE"]




    def determine_home(matchup):
        if '@' in matchup:
            return 0
        elif 'vs' in matchup:
            return 1
        else:
            return None
    # Filter the DataFrame to keep only the desired columns
    df_player_game_logs_filtered = df_player_game_logs[columns_to_keep]
    #keep onyl the opp name and get home/away
    df_player_game_logs_filtered['HOME'] = df_player_game_logs_filtered['MATCHUP'].apply(determine_home)
    df_player_game_logs_filtered['MATCHUP'] = df_player_game_logs_filtered['MATCHUP'].str[-3:]

    #df_player_game_logs_filtered = df_player_game_logs_filtered[df_player_game_logs_filtered['MIN_RANK'] >= 10000]

    # Print the first 10 rows of the filtered DataFrame
    pd.set_option('display.max_columns', None)
    #df_player_game_logs_filtered.drop(columns=['MIN_RANK'], inplace=True)

    df_player_game_logs_filtered.to_csv('filteredLog_data.csv', index=False)


    grouped_df = df_player_game_logs_filtered.groupby('PLAYER_NAME')

    # Iterate over the groups and concatenate the data for each player
    organized_df = pd.concat([group for _, group in grouped_df])
    ##cut off the dates
    organized_df['GAME_DATE'] = organized_df['GAME_DATE'].str[:10]

    organized_df.to_csv('organized_data.csv', index=False)

    organized_df = addAdvanced()
    

    
    team_id_map = {
    'ATL': '1610612737',
    'BOS': '1610612738',
    'BKN': '1610612751',
    'CHA': '1610612766',
    'CHI': '1610612741',
    'CLE': '1610612739',
    'DAL': '1610612742',
    'DEN': '1610612743',
    'DET': '1610612765',
    'GSW': '1610612744',
    'HOU': '1610612745',
    'IND': '1610612754',
    'LAC': '1610612746',
    'LAL': '1610612747',
    'MEM': '1610612763',
    'MIA': '1610612748',
    'MIL': '1610612749',
    'MIN': '1610612750',
    'NOP': '1610612740',
    'NYK': '1610612752',
    'OKC': '1610612760',
    'ORL': '1610612753',
    'PHI': '1610612755',
    'PHX': '1610612756',
    'POR': '1610612757',
    'SAC': '1610612758',
    'SAS': '1610612759',
    'TOR': '1610612761',
    'UTA': '1610612762',
    'WAS': '1610612764'
    }

    # Create a new DataFrame to store the mapped values
    mapped_df = organized_df.copy()

    # Map team abbreviations to team IDs
    mapped_df['TEAM_ID'] = mapped_df['TEAM_ABBREVIATION'].map(team_id_map)

    # Map matchup names to team IDs
    mapped_df['MATCHUP_ID'] = mapped_df['MATCHUP'].map(team_id_map)

    # Drop the original team abbreviation and matchup columns
    mapped_df.drop(['TEAM_ABBREVIATION', 'MATCHUP'], axis=1, inplace=True)
    if not backup:
        mapped_df.to_csv("NBA_DF.csv", index=False)
    return mapped_df

    

def subtract_one_year(year_range):
    # Split the input string into two parts using '-'
    start_year, end_year = year_range.split('-')

    # Convert the years to integers and subtract 1
    start_year = int(start_year) - 1
    end_year = int(end_year) - 1

    # Format the years back into the desired format and return
    return f"{start_year}-{str(end_year).zfill(2)}"


def getPastData(season):
    pastSzn = subtract_one_year(season)
    print("past szn calculated is", pastSzn)
    season_type = 'Regular Season'

    df23 = getNBADF(season, season_type)
    #df23 = pd.read_csv("NBA_DF.csv")
    df22 = getNBADF(pastSzn, season_type, season_segment = 'Post All-Star', backup = True)
    combined_df = pd.concat([df23, df22], ignore_index=True)
    combined_df_sorted = combined_df.sort_values(by='GAME_DATE')
    combined_df.to_csv("combinded_data.csv")
    # Group the sorted DataFrame by player name
    grouped_df = combined_df_sorted.groupby('PLAYER_NAME')

    # Iterate over the groups and concatenate the data for each player
    organized_df = pd.concat([group for _, group in grouped_df])

    # Save the organized data to a CSV file

    player_counts = organized_df['PLAYER_ID'].value_counts()

    # Filter out players who don't meet the threshold of 30 occurrences
    valid_players = player_counts[player_counts >= 50].index
    

    # Create a new DataFrame with only the entries of players who meet the threshold
    filtered_df = organized_df[organized_df['PLAYER_ID'].isin(valid_players)]



    # Define the columns to be averaged
    columns_to_average = ['MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                        'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'AST', 'TOV', 'PF', 
                        'PTS', 'PLUS_MINUS', 'NBA_FANTASY_PTS', 'PLAYER_USG_PCT']

    # Group the DataFrame by player name
    grouped = filtered_df.groupby('PLAYER_NAME')

    # Initialize an empty DataFrame to store the results
    result_df = pd.DataFrame()

    # Iterate through each group
    for player_name, group_df in grouped:
        # Filter the group for the 2022-23 season
        group_df_current_season = group_df[group_df['SEASON_YEAR'] == season]
        
        # Shift the DataFrame by one row to exclude the current row
        shifted_group_df = group_df_current_season[columns_to_average].shift(1)
        
        # Compute rolling averages for the shifted DataFrame
        rolling_avg_current_season = shifted_group_df.rolling(window=15, min_periods=1).mean()
        
        # If there are fewer than 15 games in the current season, look into the previous season
        if len(rolling_avg_current_season) < 15:
            # Filter the group for the 2021-22 season
            group_df_prev_season = group_df[group_df['SEASON_YEAR'] == pastSzn]
            
            # Shift the DataFrame by one row to exclude the current row
            shifted_group_df_prev_season = group_df_prev_season[columns_to_average].shift(1)
            
            # Compute rolling averages for the shifted DataFrame of the previous season
            rolling_avg_prev_season = shifted_group_df_prev_season.rolling(window=15, min_periods=1).mean()
            
            # Concatenate rolling averages from both seasons
            rolling_avg = pd.concat([rolling_avg_prev_season, rolling_avg_current_season])
        else:
            # Use rolling averages from the current season only
            rolling_avg = rolling_avg_current_season
        
        # Merge rolling averages with the original DataFrame
        group_df[columns_to_average] = rolling_avg
        result_df = pd.concat([result_df, group_df])
            


    final_df = pd.concat([result_df, filtered_df['PTS'].rename('EPTS')], axis=1)


    #drop the 2021 years that were for filler data
    final_df = final_df[final_df['SEASON_YEAR'] != pastSzn]

    #delete unwanted data and rename
    final_df = final_df.drop(columns=["SEASON_YEAR", "WL", "PLAYER_NAME", "MIN_RANK", "MIN_RANK", "GAME_DATE", "GAME_ID"])


    columns_to_rename = ['MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                        'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'AST', 'TOV', 'PF', 
                        'PTS', 'PLUS_MINUS', 'NBA_FANTASY_PTS']

    # Rename columns by adding "L25" to the beginning of each name
    new_columns = ['L15' + col if col in columns_to_rename else col for col in final_df.columns]

    # Rename columns in the DataFrame
    final_df.columns = new_columns

    #delete empty rows that couldnt be averaged (rookies)
    final_df = final_df.dropna(subset=['L15MIN'])
    print('exitng here')
    return final_df
    

def getMatchup(teamId):
    # Get today's date

    today_date = datetime.now().strftime('%Y-%m-%d')

    # Fetch today's games
    scoreboard = Scoreboard(game_date=today_date)
    games = scoreboard.game_header.get_data_frame()
    selected_columns = games.loc[:, ['HOME_TEAM_ID', 'VISITOR_TEAM_ID']]
    # Print the games happening today
    

    teams_vs_opponents = {}

    # Iterate through each row in the DataFrame
    for index, row in selected_columns.iterrows():
        # Get the home team and visitor team IDs for each game
        home_team_id = row['HOME_TEAM_ID']
        visitor_team_id = row['VISITOR_TEAM_ID']
        
        # Assume you have a function team_id_to_name() that converts team IDs to team names
        
        
        # Add the teams and their opponents to the dictionary
        teams_vs_opponents[home_team_id] = visitor_team_id
        teams_vs_opponents[visitor_team_id] = home_team_id

    return teams_vs_opponents[teamId]



def getL15(season, players):
    
    
    combined_df = pd.read_csv('NBA_DF.csv')
    combined_df_sorted = combined_df.sort_values(by='GAME_DATE')
    # Group the sorted DataFrame by player name
    grouped_df = combined_df_sorted.groupby('PLAYER_NAME')

    # Iterate over the groups and concatenate the data for each player
    organized_df = pd.concat([group for _, group in grouped_df])
    
    # Save the organized data to a CSV file

    player_counts = organized_df['PLAYER_ID'].value_counts()
    
    # Filter out players who don't meet the threshold of 50 GP
    valid_players = player_counts[player_counts >= 50].index
    
    #Create a new DataFrame with only the entries of players who meet the threshold
    filtered_df_GP = organized_df[organized_df['PLAYER_ID'].isin(valid_players)]

    filtered_df = filtered_df_GP[filtered_df_GP['PLAYER_ID'].isin(players)]
    # Define the columns to be averaged
    columns_to_average = ['MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                        'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'AST', 'TOV', 'PF', 
                        'PTS', 'PLUS_MINUS', 'NBA_FANTASY_PTS',"PLAYER_USG_PCT","TEAM_DEF_RATING","TEAM_PACE"]

    # Group the DataFrame by player name
    grouped = filtered_df.groupby('PLAYER_NAME')
    
    # Initialize an empty DataFrame to store the results
    result_df = pd.DataFrame()

    for player_id, group_df in grouped:
        # Filter the group for the 2022-23 season
        group_df = group_df[group_df['SEASON_YEAR'] == season]
        
        # Compute rolling averages for each player
        rolling_avg = group_df[columns_to_average].rolling(window=15, min_periods=1).mean()
        
        # Merge rolling averages with the original DataFrame
        group_df[columns_to_average] = rolling_avg
        result_df = pd.concat([result_df, group_df])

    final_df = pd.concat([result_df, filtered_df['PTS'].rename('EPTS')], axis=1)
    final_df = final_df.groupby('PLAYER_NAME').tail(1)

    #delete unwanted data and rename

    #*********************#
    final_df = final_df.drop(columns=["SEASON_YEAR", "WL", "PLAYER_NAME", "MIN_RANK", "MIN_RANK", "GAME_DATE", "GAME_ID","EPTS"])


    columns_to_rename = ['MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                        'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'AST', 'TOV', 'PF', 
                        'PTS', 'PLUS_MINUS', 'NBA_FANTASY_PTS']

    # Rename columns by adding "L25" to the beginning of each name
    new_columns = ['L15' + col if col in columns_to_rename else col for col in final_df.columns]

    # Rename columns in the DataFrame
    final_df.columns = new_columns

    #delete empty rows that couldnt be averaged (rookies)
    final_df = final_df.dropna(subset=['L15MIN'])

    for index, row in final_df.iterrows():
        team_id = row['TEAM_ID']
        matchup = getMatchup(team_id)
        final_df.at[index, 'MATCHUP'] = matchup

    def get_team_metrics(season, season_type):
        # Retrieve team metrics
        team_metrics = TeamEstimatedMetrics(
            season=season,
            season_type=season_type
        )
        data = team_metrics.get_data_frames()[0]
        return data
    
    
    season_type = 'Regular Season'
    team_metrics_data = get_team_metrics(season, season_type)
    for index, row in final_df.iterrows():
        matchup_id = row['MATCHUP']
        
        # geth specific data based off of the matchup_id
        
        team_data = team_metrics_data[team_metrics_data['TEAM_ID'] == matchup_id]
        
        # Update the values in final_df
        final_df.at[index, 'TEAM_DEF_RATING'] = team_data['E_DEF_RATING'].values[0]
        final_df.at[index, 'TEAM_PACE'] = team_data['E_PACE'].values[0]


    final_df['MATCHUP'], final_df['MATCHUP_ID'] = final_df['MATCHUP_ID'], final_df['MATCHUP']

    # Delete the MATCHUP column
    final_df.drop(columns=['MATCHUP'], inplace=True)

    final_df.to_csv('L15_data.csv', index=False)
    print('exitng here')
    return final_df










