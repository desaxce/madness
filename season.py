from game import Game
from tournament import Tournament


class Season:
    def __init__(self, year: int, day_zero: str, regular_season_games: [Game], tournament_games: [Game], region_w: str, region_x: str, region_y: str, region_z: str):
        self.regular_season = RegularSeason(year, day_zero, regular_season_games)
        self.tournament = Tournament(tournament_games, region_w, region_x, region_y, region_z)

    def get_match_ups_features(self):
        X = []
        y = []
        for game in self.tournament.tournament_games:
            w_record = self.regular_season.get_record(game.w_team_id)
            l_record = self.regular_season.get_record(game.l_team_id)
            X.append([w_record, l_record])
            y.append(1)
            X.append([l_record, w_record])
            y.append(0)
        return X, y


class RegularSeason:
    def __init__(self, year: int, day_zero: str, regular_season_games: [Game]):
        self.year = year
        self.day_zero = day_zero
        self.regular_season_games = regular_season_games

    def get_record(self, team_id: str) -> float:
        won_games = 0
        lost_games = 0
        for game in self.regular_season_games:
            if game.w_team_id == team_id:
                won_games += 1
            elif game.l_team_id == team_id:
                lost_games += 1
        return won_games / (won_games + lost_games)