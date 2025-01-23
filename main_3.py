import argparse
import os
from datetime import datetime, timedelta
import pandas as pd
import tensorflow as tf
from colorama import Fore, Style
import requests

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.Predict import NN_Runner, XGBoost_Runner
from src.Utils.Dictionaries import team_index_current
from src.Utils.tools import create_todays_games_from_odds, get_json_data, to_data_frame, get_todays_games_json, create_todays_games

todays_games_url = 'https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/2024/scores/00_todays_scores.json'
data_url = 'https://stats.nba.com/stats/leaguedashteamstats?' \
           'Conference=&DateFrom=&DateTo=&Division=&GameScope=&' \
           'GameSegment=&LastNGames=0&LeagueID=00&Location=&' \
           'MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&' \
           'PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&' \
           'PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&' \
           'Season=2024-25&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&' \
           'StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='

def get_odds(sportsbook):
    """Get odds from the-odds-api.com"""
    base_url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
    api_key = "7c420299a4eb936dd39aebecc3d54fb8"
    if not api_key:
        print(Fore.RED + "No API key found. Please set ODDS_API_KEY environment variable" + Style.RESET_ALL)
        return None

    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,totals",
        "oddsFormat": "american"
    }
    
    try:
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print(Fore.RED + f"API request failed with status code: {response.status_code}" + Style.RESET_ALL)
            return None
            
        games_data = response.json()
        if not games_data:
            print(Fore.RED + "No games data received from API" + Style.RESET_ALL)
            return None
        
        formatted_odds = {}
        for game in games_data:
            home_team = game['home_team']
            away_team = game['away_team']
            
            bookmaker_data = next((b for b in game['bookmakers'] if b['key'] == sportsbook), None)
            if not bookmaker_data:
                continue

            try:
                h2h_market = next(m for m in bookmaker_data['markets'] if m['key'] == 'h2h')
                totals_market = next(m for m in bookmaker_data['markets'] if m['key'] == 'totals')
                
                home_odds = next(o for o in h2h_market['outcomes'] if o['name'] == home_team)['price']
                away_odds = next(o for o in h2h_market['outcomes'] if o['name'] == away_team)['price']
                over_under = float(next(o for o in totals_market['outcomes'] if o['name'] == 'Over')['point'])
                
                formatted_odds[f"{home_team}:{away_team}"] = {
                    'under_over_odds': over_under,
                    home_team: {'money_line_odds': home_odds},
                    away_team: {'money_line_odds': away_odds}
                }
            except Exception as e:
                print(Fore.YELLOW + f"Warning: Could not process odds for {home_team} vs {away_team}: {str(e)}" + Style.RESET_ALL)
                continue
        
        if not formatted_odds:
            print(Fore.RED + "No valid odds data could be formatted" + Style.RESET_ALL)
            return None
            
        return formatted_odds
        
    except Exception as e:
        print(Fore.RED + f"Error getting odds: {str(e)}" + Style.RESET_ALL)
        return None

def createTodaysGames(games, df, odds):
    """Create today's games data with rest days calculation"""
    if not games:
        raise ValueError("No games provided")
        
    match_data = []
    todays_games_uo = []
    home_team_odds = []
    away_team_odds = []
    home_team_days_rest = []
    away_team_days_rest = []

    try:
        schedule_df = pd.read_csv('Data/nba-2024-UTC.csv', parse_dates=['Date'], date_format='%d/%m/%Y %H:%M')
    except Exception as e:
        print(Fore.RED + f"Error reading schedule data: {str(e)}" + Style.RESET_ALL)
        raise

    for game in games:
        try:
            home_team, away_team = game[0], game[1]
            if home_team not in team_index_current or away_team not in team_index_current:
                print(Fore.YELLOW + f"Warning: Skipping {home_team} vs {away_team} - team not found in index" + Style.RESET_ALL)
                continue

            if odds is not None:
                game_key = f"{home_team}:{away_team}"
                if game_key not in odds:
                    print(Fore.YELLOW + f"Warning: No odds found for {game_key}" + Style.RESET_ALL)
                    continue
                    
                game_odds = odds[game_key]
                todays_games_uo.append(game_odds['under_over_odds'])
                home_team_odds.append(game_odds[home_team]['money_line_odds'])
                away_team_odds.append(game_odds[away_team]['money_line_odds'])
            else:
                todays_games_uo.append(float(input(f"{home_team} vs {away_team} over/under: ")))
                home_team_odds.append(float(input(f"{home_team} odds: ")))
                away_team_odds.append(float(input(f"{away_team} odds: ")))

            # Calculate rest days
            home_games = schedule_df[(schedule_df['Home Team'] == home_team) | (schedule_df['Away Team'] == home_team)]
            away_games = schedule_df[(schedule_df['Home Team'] == away_team) | (schedule_df['Away Team'] == away_team)]
            
            today = datetime.today()
            
            previous_home_games = home_games[home_games['Date'] <= today].sort_values('Date', ascending=False)
            previous_away_games = away_games[away_games['Date'] <= today].sort_values('Date', ascending=False)
            
            home_days_off = timedelta(days=7)  # Default to 7 days if no previous games
            away_days_off = timedelta(days=7)
            
            if not previous_home_games.empty:
                last_home_date = previous_home_games.iloc[0]['Date']
                home_days_off = timedelta(days=1) + today - last_home_date
                
            if not previous_away_games.empty:
                last_away_date = previous_away_games.iloc[0]['Date']
                away_days_off = timedelta(days=1) + today - last_away_date

            home_team_days_rest.append(home_days_off.days)
            away_team_days_rest.append(away_days_off.days)

            # Create game stats
            home_team_series = df.iloc[team_index_current[home_team]]
            away_team_series = df.iloc[team_index_current[away_team]]
            stats = pd.concat([home_team_series, away_team_series])
            stats['Days-Rest-Home'] = home_days_off.days
            stats['Days-Rest-Away'] = away_days_off.days
            match_data.append(stats)

        except Exception as e:
            print(Fore.RED + f"Error processing game {home_team} vs {away_team}: {str(e)}" + Style.RESET_ALL)
            continue

    if not match_data:
        raise ValueError("No valid games could be processed")

    games_data_frame = pd.concat(match_data, ignore_index=True, axis=1)
    games_data_frame = games_data_frame.T

    frame_ml = games_data_frame.drop(columns=['TEAM_ID', 'TEAM_NAME'])
    data = frame_ml.values
    data = data.astype(float)

    return data, todays_games_uo, frame_ml, home_team_odds, away_team_odds

def main():
    odds = None
    if args.odds:
        odds = get_odds(sportsbook=args.odds)
        if odds is None:
            print(Fore.RED + "Failed to fetch odds data. Falling back to manual input mode." + Style.RESET_ALL)
            # Fall back to using the regular games fetching method
            data = get_todays_games_json(todays_games_url)
            games = create_todays_games(data)
        else:
            try:
                games = create_todays_games_from_odds(odds)
                if not games:  # Check if games list is empty
                    print(Fore.RED + "No games found in odds data." + Style.RESET_ALL)
                    return
                    
                # Validate that odds data matches games
                first_game_key = f"{games[0][0]}:{games[0][1]}"
                if first_game_key not in odds:
                    print(Fore.RED + f"Games list not up to date for today's games! ({first_game_key} not found in odds data)" + Style.RESET_ALL)
                    print("Falling back to manual input mode.")
                    odds = None
                    data = get_todays_games_json(todays_games_url)
                    games = create_todays_games(data)
                else:
                    print(f"------------------{args.odds} odds data------------------")
                    for g in odds.keys():
                        home_team, away_team = g.split(":")
                        print(f"{away_team} ({odds[g][away_team]['money_line_odds']}) @ {home_team} ({odds[g][home_team]['money_line_odds']})")
            except Exception as e:
                print(Fore.RED + f"Error processing odds data: {str(e)}" + Style.RESET_ALL)
                print("Falling back to manual input mode.")
                odds = None
                data = get_todays_games_json(todays_games_url)
                games = create_todays_games(data)
    else:
        data = get_todays_games_json(todays_games_url)
        games = create_todays_games(data)

    try:
        data = get_json_data(data_url)
        if data is None:
            print(Fore.RED + "Failed to fetch NBA stats data" + Style.RESET_ALL)
            return
            
        df = to_data_frame(data)
        data, todays_games_uo, frame_ml, home_team_odds, away_team_odds = createTodaysGames(games, df, odds)
        
        if args.nn:
            print("------------Neural Network Model Predictions-----------")
            normalized_data = tf.keras.utils.normalize(data, axis=1)
            NN_Runner.nn_runner(normalized_data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
            print("-------------------------------------------------------")
        if args.xgb:
            print("---------------XGBoost Model Predictions---------------")
            XGBoost_Runner.xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
            print("-------------------------------------------------------")
        if args.A:
            print("---------------XGBoost Model Predictions---------------")
            XGBoost_Runner.xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
            print("-------------------------------------------------------")
            normalized_data = tf.keras.utils.normalize(data, axis=1)
            print("------------Neural Network Model Predictions-----------")
            NN_Runner.nn_runner(normalized_data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
            print("-------------------------------------------------------")
            
    except Exception as e:
        print(Fore.RED + f"Error during model execution: {str(e)}" + Style.RESET_ALL)
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NBA Game Prediction Model')
    parser.add_argument('-xgb', action='store_true', help='Run with XGBoost Model')
    parser.add_argument('-nn', action='store_true', help='Run with Neural Network Model')
    parser.add_argument('-A', action='store_true', help='Run all Models')
    parser.add_argument('-odds', help='Sportsbook to fetch from. (fanduel, draftkings, betmgm, pointsbet, caesars)')
    parser.add_argument('-kc', action='store_true', help='Calculates percentage of bankroll to bet based on model edge')
    args = parser.parse_args()
    
    try:
        main()
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\nProgram terminated by user" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"Fatal error: {str(e)}" + Style.RESET_ALL)