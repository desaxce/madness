import csv, logging

from game import Game
from season import Season
from teams import Team

from sklearn.neural_network import MLPClassifier


class Parser:
    def __init__(self):
        self.path = 'resources/'
        self.logger = self._get_logger()

    @staticmethod
    def _get_logger():
        return logging.getLogger(__name__)

    def parse(self):
        teams = self.parse_teams()
        regular_seasons_games = self.parse_regular_seasons_games()
        tournaments_games = self.parse_tournaments_games()
        seasons = self.parse_seasons(regular_seasons_games, tournaments_games, teams)
        return seasons, teams

    def parse_teams(self) -> [Team]:
        teams: [Team] = []
        with open(self.path + 'MTeams.csv') as teams_csv:
            csv_reader = csv.reader(teams_csv, delimiter=',')
            line_count: int = 0
            for row in csv_reader:
                if line_count == 0:
                    self.logger.info(f'Column names are {", ".join(row)}')
                else:
                    team_id: int = int(row[0])
                    team_name: str = row[1]
                    first_d1_season: int = int(row[2])
                    last_d1_season: int = int(row[3])
                    teams.append(Team(team_id, team_name, first_d1_season, last_d1_season))
                line_count += 1
            self.logger.info(f'Processed {line_count - 1} teams.')
        return teams

    def parse_seasons(self, regular_seasons_games: [Game], tournaments_games: [Game], teams: [Team]):
        seasons: dict[int, Season] = {}
        with open(self.path + 'MSeasons.csv') as seasons_csv:
            csv_reader = csv.reader(seasons_csv, delimiter=',')
            line_count: int = 0
            for row in csv_reader:
                if line_count == 0:
                    self.logger.info(f'Column names are {", ".join(row)}')
                else:
                    year = int(row[0])
                    day_zero = row[1]
                    region_w = row[2]
                    region_x = row[3]
                    region_y = row[4]
                    region_z = row[5]
                    if year not in seasons:
                        seasons[year] = []
                    regular_season_games = regular_seasons_games[year]
                    tournament_games = []
                    if year in tournaments_games:
                        tournament_games = tournaments_games[year]
                    seasons[year] = Season(year, day_zero, regular_season_games,
                                           tournament_games, region_w, region_x, region_y, region_z,
                                           teams)
                line_count += 1
            self.logger.info(f'Processed {line_count - 1} seasons.')
        return seasons

    def parse_regular_seasons_games(self):
        games: dict[int, Game] = {}
        with open(self.path + 'MRegularSeasonCompactResults.csv') as seasons_csv:
            csv_reader = csv.reader(seasons_csv, delimiter=',')
            line_count: int = 0
            for row in csv_reader:
                if line_count == 0:
                    self.logger.info(f'Column names are {", ".join(row)}')
                else:
                    year: int = int(row[0])
                    day_num: int = int(row[1])
                    w_team_id: int = int(row[2])
                    w_score: int = int(row[3])
                    l_team_id: int = int(row[4])
                    l_score: int = int(row[5])
                    w_loc: str = row[6]
                    num_ot: int = int(row[7])
                    if year not in games:
                        games[year] = []
                    games[year].append(Game(year, day_num, w_team_id, w_score, l_team_id, l_score, w_loc, num_ot))
                line_count += 1
            self.logger.info(f'Processed {line_count - 1} games.')
        return games

    def parse_tournaments_games(self):
        games: dict[int, Game] = {}
        with open(self.path + 'MNCAATourneyCompactResults.csv') as seasons_csv:
            csv_reader = csv.reader(seasons_csv, delimiter=',')
            line_count: int = 0
            for row in csv_reader:
                if line_count == 0:
                    self.logger.info(f'Column names are {", ".join(row)}')
                else:
                    year: int = int(row[0])
                    day_num: int = int(row[1])
                    w_team_id: int = int(row[2])
                    w_score: int = int(row[3])
                    l_team_id: int = int(row[4])
                    l_score: int = int(row[5])
                    w_loc: str = row[6]
                    num_ot: int = int(row[7])
                    if year not in games:
                        games[year] = []
                    games[year].append(Game(year, day_num, w_team_id, w_score, l_team_id, l_score, w_loc, num_ot))
                line_count += 1
            self.logger.info(f'Processed {line_count - 1} games.')
        return games
