from nba_api.stats.endpoints import PlayerGameLogs, TeamEstimatedMetrics, BoxScoreAdvancedV2, TeamDashboardByGeneralSplits, Scoreboard
from nba_api.stats.static import players
import pandas as pd
from datetime import datetime
from getData import getL15
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

from nba_api.stats.endpoints import BoxScoreAdvancedV2

def get_player_id(player_name):
    
    matching_players = players.find_players_by_full_name(player_name)
    if len(matching_players) == 0:
        return None
    elif len(matching_players) > 1:
        print("Multiple players found with the same name. Please specify.")
        return None
    else:
        return matching_players[0]['id']
    
players_ = ['Alperen Sengun', 'Jalen Green', 'Domantas Sabonis', 'Damian Lillard','Malik Beasley', 'Brook Lopez', 'Giannis Antetokounmpo']
player_ids = [get_player_id(player_name) for player_name in players_]
# df = getL15('2023-24', player_ids)
# df.to_csv('last15Test.csv', index = False)



def get_team_metrics(season, season_type):
        # Retrieve team metrics
        team_metrics = TeamEstimatedMetrics(
            season=season,
            season_type=season_type
        )
        data = team_metrics.get_data_frames()[0]
        return data
    
    
season_type = 'Regular Season'
team_metrics_data = get_team_metrics('2023-24', season_type)
sacramento_kings_stats = team_metrics_data[team_metrics_data['TEAM_NAME'] == 'Sacramento Kings']
sacramento_defense_rating = sacramento_kings_stats['E_DEF_RATING'].values[0]
print(sacramento_defense_rating)

#get shot metrics
#for each team add https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/defensehub.md