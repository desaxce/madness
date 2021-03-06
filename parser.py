import csv, logging
from collections import defaultdict
from typing import Dict

from game import Game
from season import Season
from teams import Team
from seed import Seed


class Parser:
    def __init__(self, gender: str = "W"):
        self.path = 'resources/'
        self.gender = gender
        self.logger = self._get_logger()

    @staticmethod
    def _get_logger():
        return logging.getLogger(__name__)

    def parse(self):
        teams = self.parse_teams()
        regular_seasons_games = self.parse_regular_seasons_games()
        # print(regular_seasons_games[2022])
        tournaments_games = self.parse_tournaments_games()
        seasons_seeds = self.parse_seeds()
        seasons_rankings = defaultdict(lambda: {})

        if self.gender == "M":
            seasons_rankings = self.parse_rankings() #defaultdict(lambda: {})
        seasons = self.parse_seasons(regular_seasons_games, tournaments_games, teams, seasons_seeds, seasons_rankings)
        return seasons, teams

    def parse_rankings(self) -> Dict:
        seasons_rankings: Dict = {}
        with open(self.path + self.gender + 'MasseyOrdinals.csv') as rankings_csv:
            csv_reader = csv.reader(rankings_csv, delimiter=',')
            line_count: int = 0
            for row in csv_reader:
                if line_count == 0:
                    self.logger.info(f'Column names are {", ".join(row)}')
                else:
                    year: int = int(row[0])
                    day_num: int = int(row[1])
                    system_name: str = row[2]
                    team_id: int = int(row[3])
                    rank: int = int(row[4])

                    # The final pre-tournament rankings each year have a RankingDayNum of 133
                    if day_num == 133:
                        if year not in seasons_rankings:
                            seasons_rankings[year] = {}

                        if system_name not in seasons_rankings[year]:
                            seasons_rankings[year][system_name] = {}

                        seasons_rankings[year][system_name][team_id] = rank
                line_count += 1
            self.logger.info(f'Processed {line_count - 1} ranking rows.')
        return seasons_rankings

    def parse_seeds(self) -> Dict:
        seasons_seeds: Dict = {}
        with open(self.path + self.gender + 'NCAATourneySeeds.csv') as seeds_csv:
            csv_reader = csv.reader(seeds_csv, delimiter=',')
            line_count: int = 0
            for row in csv_reader:
                if line_count == 0:
                    self.logger.info(f'Column names are {", ".join(row)}')
                else:
                    year: int = int(row[0])
                    seed: str = row[1]
                    team_id: int = int(row[2])
                    if year not in seasons_seeds:
                        seasons_seeds[year] = []
                    seasons_seeds[year].append(Seed(year, seed, team_id))
                line_count += 1
            self.logger.info(f'Processed {line_count - 1} seeds.')
        return seasons_seeds

    def parse_teams(self) -> [Team]:
        teams: [Team] = []
        with open(self.path + self.gender + 'Teams.csv') as teams_csv:
            csv_reader = csv.reader(teams_csv, delimiter=',')
            line_count: int = 0
            for row in csv_reader:
                if line_count == 0:
                    self.logger.info(f'Column names are {", ".join(row)}')
                else:
                    team_id: int = int(row[0])
                    team_name: str = row[1]
                    first_d1_season = 0
                    last_d1_season = 0
                    if self.gender == "M":
                        first_d1_season: int = int(row[2])
                        last_d1_season: int = int(row[3])
                    teams.append(Team(team_id, team_name, first_d1_season, last_d1_season))
                line_count += 1
            self.logger.info(f'Processed {line_count - 1} teams.')
        return teams

    def parse_seasons(self, regular_seasons_games: Dict, tournaments_games: Dict, teams: [Team],
                      seasons_seeds: Dict, seasons_rankings: Dict):
        seasons: Dict[int, Season] = {}
        with open(self.path + self.gender + 'Seasons.csv') as seasons_csv:
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

                    # if year <= 2002:
                    #     continue

                    if year not in seasons:
                        seasons[year] = []

                    # There is a regular season for every year.
                    regular_season_games = regular_seasons_games[year]

                    tournament_games = []
                    seeds: [Seed] = []
                    rankings: Dict = {}

                    # Specific check to avoid the 2019-2020 season which didn't see the NCAA tournament take place.
                    if year not in [2020, 2022]:
                        tournament_games = tournaments_games[year]

                    if year not in [2020]:
                        seeds = seasons_seeds[year]

                    # Ranks only available starting with the 2002-03 season.
                    if self.gender == "M" and year not in [2020] and year >= 2003:
                        rankings = seasons_rankings[year]

                    if year not in [2020]:  # Skip 2020, no use of that season.
                        seasons[year] = Season(year, day_zero, regular_season_games, rankings,
                                               tournament_games, region_w, region_x, region_y, region_z, seeds,
                                               teams)
                line_count += 1
            self.logger.info(f'Processed {line_count - 1} seasons.')
        return seasons

    def parse_regular_seasons_games(self):
        games: Dict = {}
        with open(self.path + self.gender + 'RegularSeasonCompactResults.csv') as seasons_csv:
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

                    # w_fgm: int = int(row[8])
                    # w_fga: int = int(row[9])
                    # w_fgm3: int = int(row[10])
                    # w_fga3: int = int(row[11])
                    # w_ftm: int = int(row[12])
                    # w_fta: int = int(row[13])
                    # w_or: int = int(row[14])
                    # w_dr: int = int(row[15])
                    # w_ast: int = int(row[16])
                    # w_to: int = int(row[17])
                    # w_stl: int = int(row[18])
                    # w_blk: int = int(row[19])
                    # w_pf: int = int(row[20])
                    #
                    # l_fgm: int = int(row[21])
                    # l_fga: int = int(row[22])
                    # l_fgm3: int = int(row[23])
                    # l_fga3: int = int(row[24])
                    # l_ftm: int = int(row[25])
                    # l_fta: int = int(row[26])
                    # l_or: int = int(row[27])
                    # l_dr: int = int(row[28])
                    # l_ast: int = int(row[29])
                    # l_to: int = int(row[30])
                    # l_stl: int = int(row[31])
                    # l_blk: int = int(row[32])
                    # l_pf: int = int(row[33])
                    if year not in games:
                        games[year] = []
                    games[year].append(
                        Game(year, day_num, w_team_id, w_score, l_team_id, l_score, w_loc, num_ot))
                    """, w_fgm, w_fga, w_fgm3,
                             w_fga3,
                             w_ftm, w_fta, w_or, w_dr, w_ast, w_to, w_stl, w_blk, w_pf, l_fgm, l_fga, l_fgm3, l_fga3,
                             l_ftm, l_fta, l_or, l_dr, l_ast, l_to, l_stl, l_blk, l_pf"""
                line_count += 1
            self.logger.info(f'Processed {line_count - 1} games.')
        return games

    def parse_tournaments_games(self):
        games: Dict = {}
        with open(self.path + self.gender +  'NCAATourneyCompactResults.csv') as seasons_csv:
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

                    # w_fgm: int = int(row[8])
                    # w_fga: int = int(row[9])
                    # w_fgm3: int = int(row[10])
                    # w_fga3: int = int(row[11])
                    # w_ftm: int = int(row[12])
                    # w_fta: int = int(row[13])
                    # w_or: int = int(row[14])
                    # w_dr: int = int(row[15])
                    # w_ast: int = int(row[16])
                    # w_to: int = int(row[17])
                    # w_stl: int = int(row[18])
                    # w_blk: int = int(row[19])
                    # w_pf: int = int(row[20])
                    #
                    # l_fgm: int = int(row[21])
                    # l_fga: int = int(row[22])
                    # l_fgm3: int = int(row[23])
                    # l_fga3: int = int(row[24])
                    # l_ftm: int = int(row[25])
                    # l_fta: int = int(row[26])
                    # l_or: int = int(row[27])
                    # l_dr: int = int(row[28])
                    # l_ast: int = int(row[29])
                    # l_to: int = int(row[30])
                    # l_stl: int = int(row[31])
                    # l_blk: int = int(row[32])
                    # l_pf: int = int(row[33])
                    if year not in games:
                        games[year] = []
                    games[year].append(
                        Game(year, day_num, w_team_id, w_score, l_team_id, l_score, w_loc, num_ot))
                    """, w_fgm, w_fga, w_fgm3,
                             w_fga3,
                             w_ftm, w_fta, w_or, w_dr, w_ast, w_to, w_stl, w_blk, w_pf, l_fgm, l_fga, l_fgm3, l_fga3,
                             l_ftm, l_fta, l_or, l_dr, l_ast, l_to, l_stl, l_blk, l_pf"""
                line_count += 1
            self.logger.info(f'Processed {line_count - 1} games.')
        return games
