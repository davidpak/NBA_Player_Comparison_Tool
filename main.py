from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.endpoints import commonplayerinfo
from nba_api.stats.static import players
from nba_api.stats.endpoints import alltimeleadersgrids
from sklearn.metrics.pairwise import cosine_similarity


import pandas as pd
from scipy.spatial import distance


def compare_two_players(player1_name, player2_name):
    # Find player IDs using the nba_api
    player1 = players.find_players_by_full_name(player1_name)
    player2 = players.find_players_by_full_name(player2_name)

    # If both players are found
    if player1 and player2:
        player1_id = player1[0]['id']
        player2_id = player2[0]['id']

        # Formally Define Metrics
        metrics = [
            'PTS', 'FGM', 'FGA', 'FG_PCT', 'FTM', 'FTA', 'FT_PCT',
            'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV',
            'FG3M', 'FG3A', 'FG3_PCT', 'HEIGHT', 'WEIGHT', 'POSITION'
        ]

        # Retrieve career statistics for player 1
        player1_stats = playercareerstats.PlayerCareerStats(per_mode36='PerGame', player_id=player1_id)
        player2_stats = playercareerstats.PlayerCareerStats(per_mode36='PerGame', player_id=player2_id)
        player1_reg_season_stats = player1_stats.career_totals_regular_season.get_data_frame()
        player2_reg_season_stats = player2_stats.career_totals_regular_season.get_data_frame()

        # Retrieve player1 info
        player1_info_init = commonplayerinfo.CommonPlayerInfo(player_id=player1_id)
        player2_info_init = commonplayerinfo.CommonPlayerInfo(player_id=player2_id)
        player1_info = player1_info_init.get_data_frames()[0]
        player2_info = player2_info_init.get_data_frames()[0]

        # Extract stats for player 1
        player1_means_df = create_df(player1_reg_season_stats, player1_info, metrics)
        player2_means_df = create_df(player2_reg_season_stats, player2_info, metrics)

        # Print the data frames
        print(player1_means_df)
        print("\n")
        print(player2_means_df)

        # Get the min and max values of all metrics for data normalization
        max_values, min_values = get_min_max(metrics)

        print("Max Values:")
        print(max_values)
        print("\nMin Values:")
        print(min_values)

        # Normalize both player's metrics from 0 to 1 and print the results
        player1_normalized = normalize_metrics(max_values, min_values, player1_means_df, player1_info, metrics)
        player2_normalized = normalize_metrics(max_values, min_values, player2_means_df, player2_info, metrics)
        print(f"\n{player1_name}'s normalized metrics")
        print(player1_normalized)
        print(f"\n{player2_name}'s normalized metrics")
        print(player2_normalized)

        similarity_score = calculate_similarity_score(player1_normalized, player2_normalized, metrics)
        print(f"Similarity Score: {similarity_score:.4f}")

    else:
        print("Player not found")


def get_min_max(metrics):
    max_values = {}
    # Hard code the min values because idk how else to do it
    min_values = {
        "PTS": 3.5,
        "FGM": 1.2,
        "FGA": 3.2,
        "FG_PCT": 0.302,
        "FTM": 0.7,
        "FTA": 1,
        "FT_PCT": 0.414,
        "OREB": 0.3,
        "DREB": 0.9,
        "REB": 1.1,
        "AST": 0.5,
        "STL": 0.3,
        "BLK": 0.1,
        "TOV": 0.9,
        "FG3M": 0,
        "FG3A": 0,
        "FG3_PCT": 0.266,
        "HEIGHT": 5.99,
        "WEIGHT": 160,
        "POSITION": 1
    }

    stats = alltimeleadersgrids.AllTimeLeadersGrids(league_id='00', per_mode_simple='PerGame',
                                                    season_type='Regular Season', topx='10')

    # Obtain relevant dataframes
    pts_df = stats.pts_leaders.get_data_frame()
    fgm_df = stats.fgm_leaders.get_data_frame()
    fga_df = stats.fga_leaders.get_data_frame()
    fgp_df = stats.fg_pct_leaders.get_data_frame()
    ftm_df = stats.ftm_leaders.get_data_frame()
    fta_df = stats.fta_leaders.get_data_frame()
    ftp_df = stats.ft_pct_leaders.get_data_frame()
    oreb_df = stats.oreb_leaders.get_data_frame()
    dreb_df = stats.dreb_leaders.get_data_frame()
    reb_df = stats.reb_leaders.get_data_frame()
    ast_df = stats.ast_leaders.get_data_frame()
    stl_df = stats.stl_leaders.get_data_frame()
    blk_df = stats.blk_leaders.get_data_frame()
    tov_df = stats.tov_leaders.get_data_frame()
    fg3m_df = stats.fg3_m_leaders.get_data_frame()
    fg3a_df = stats.fg3_a_leaders.get_data_frame()
    fg3p_df = stats.fg3_pct_leaders.get_data_frame()
    height_df = pd.DataFrame()
    weight_df = pd.DataFrame()
    pos_df = pd.DataFrame()

    # Grab all maxes and mins and put into respective dictionaries
    metrics_df_list = [pts_df, fgm_df, fga_df, fgp_df, ftm_df, fta_df, ftp_df,
                       oreb_df, dreb_df, reb_df, ast_df, stl_df, blk_df, tov_df,
                       fg3m_df, fg3a_df, fg3p_df, height_df, weight_df, pos_df]

    for metric, metric_df in zip(metrics, metrics_df_list):
        # Hardcode the height, weight, and position metrics
        if metric == "HEIGHT":
            max_values[metric] = 7.4
        elif metric == "WEIGHT":
            max_values[metric] = 290
        elif metric == "POSITION":
            max_values[metric] = 5
        else:
            max_value = metric_df[metric].max()
            max_values[metric] = max_value
    return max_values, min_values


# Take all of the metrics of every player and normalize it to a value between 0 and 1
def normalize_metrics(max_values, min_values, player_means_df, player_info, metrics):
    normalized_metrics = {}
    for metric in metrics:
        # Use normalization equation to get value
        max_value = max_values.get(metric)
        min_value = min_values.get(metric)
        # Error handling to make sure that if a specific player is missing a metric, it will be set to 0
        if player_means_df.loc[player_means_df['Metric'] == metric, player_info['DISPLAY_FIRST_LAST'].values[0]].values[0] is not None:
            actual_value = float(player_means_df.loc[
                                     player_means_df['Metric'] == metric, player_info['DISPLAY_FIRST_LAST'].values[
                                         0]].values[0])
            # Since turnovers are inversely proportional to a player's effectiveness, we must use the complement
            if metric == "TOV":
                normalized_value = 1 - (actual_value - min_value) / (max_value - min_value)
            elif metric == "POSITION":
                normalized_value = actual_value
            else:
                normalized_value = (actual_value - min_value) / (max_value - min_value)
            normalized_metrics[metric] = normalized_value
        else:
            # Handle the case where the metric is not listed for the player
            print(f"{player_info['DISPLAY_FIRST_LAST'].values[0]} does not have data for {metric}. Setting to 0.")
            normalized_metrics[metric] = 0
    return normalized_metrics


def calculate_similarity_score(player1_normalized, player2_normalized, metrics):
    similarity_scores = []
    for metric in metrics:
        player1_value = player1_normalized[metric]
        player2_value = player2_normalized[metric]
        similarity_scores.append(1 - abs(player1_value - player2_value))
    return sum(similarity_scores) / len(similarity_scores)
def create_df(player_reg_season_stats, player_info, metrics):
    # Extract stats for player
    ppg_player = player_reg_season_stats['PTS'].mean()
    fgm_player = player_reg_season_stats['FGM'].mean()
    fga_player = player_reg_season_stats['FGA'].mean()
    fgp_player = player_reg_season_stats['FG_PCT'].mean()
    ftm_player = player_reg_season_stats['FTM'].mean()
    fta_player = player_reg_season_stats['FTA'].mean()
    ftp_player = player_reg_season_stats['FT_PCT'].mean()
    oreb_player = player_reg_season_stats['OREB'].mean()
    dreb_player = player_reg_season_stats['DREB'].mean()
    reb_player = player_reg_season_stats['REB'].mean()
    ast_player = player_reg_season_stats['AST'].mean()
    stl_player = player_reg_season_stats['STL'].mean()
    blk_player = player_reg_season_stats['BLK'].mean()
    tov_player = player_reg_season_stats['TOV'].mean()
    fg3m_player = player_reg_season_stats['FG3M'].mean()
    fg3a_player = player_reg_season_stats['FG3A'].mean()
    fg3p_player = player_reg_season_stats['FG3_PCT'].mean()

    # Preprocess the data for the height and position since they are formatted differently
    height_player = replace_dash_with_dot(player_info['HEIGHT'].values[0])
    weight_player = player_info['WEIGHT'].values[0]
    position_player = replace_pos_with_number(player_info['POSITION'].values[0])

    player_means_df = pd.DataFrame({
        'Metric': metrics,
        player_info['DISPLAY_FIRST_LAST'].values[0]: [
            ppg_player, fgm_player, fga_player, fgp_player,
            ftm_player, fta_player, ftp_player, oreb_player,
            dreb_player, reb_player, ast_player, stl_player,
            blk_player, tov_player, fg3m_player, fg3a_player,
            fg3p_player, height_player, weight_player, position_player
        ]
    })

    return player_means_df


def replace_dash_with_dot(height_str):
    # Replace '-' with '.' and convert to float
    return float(height_str.replace('-', '.'))


def replace_pos_with_number(position):
    if position == "Guard":
        return 1
    elif position == "Forward":
        return 3
    elif position == "Center":
        return 5


def main():
    compare_two_players("Michael Jordan", "Kawhi Leonard")


if __name__ == '__main__':
    main()