from typing import Dict

from game import Game
from classifier import Classifier, SeedsBasedClassifier
from sample import Sample
from teams import Team
from tournament import Tournament
from seed import Seed


class Season:
    """
    A season contains information about a college basketball season for division 1 teams:
    - regular season data: games;
    - tournament data: seeds and games.

    Each season also contains meta data about the division 1 teams: not only the division 1 teams for
    that season, but all teams that ever made to division 1 since the 1985 season. For the 1985-2022 span,
    there are a total of 372 division 1 teams, and the 2022 season has 358 division 1 teams. The excess of
    teams represents about 4 % of yearly division 1 teams, deemed marginal.

    Storing teams data within each season is redundant though. Because we follow an all in-memory approach
    (no databases), and we need teams meta data available at the season level, we decided to go with that
    redundancy. The impact is minimal because we have less than 40 seasons and 400 teams.

    Seasons start in late fall of year n and end with the National Collegiate Athletic Association (NCAA)
    championship played in early spring of year n+1.
    We refer to a season's year by using the year the NCAA tournament for that season was played (n+1 above).
    """
    def __init__(self, year: int, day_zero: str, regular_season_games: [Game],
                 tournament_games: [Game], region_w: str, region_x: str, region_y: str, region_z: str, seeds: [Seed],
                 teams: [Team]):
        self.regular_season: RegularSeason = RegularSeason(year, day_zero, regular_season_games)
        self.tournament: Tournament = Tournament(tournament_games, region_w, region_x, region_y, region_z, seeds)
        self.teams: [Team] = teams

    """
    Returns features and labels for the specified match-up: the return type is a tuple of two arrays, each
    of length 2:
    - features first: [[w_team_id vs. l_team_id], [l_team_id vs. w_team_id]];
    - labels second: [1, 0].
    
    The features are of two kinds:
    - either relate to a single team and its overall situation relative to all other teams;
    - or involve the relative performance of the two matched up teams.
    
    Because of the asymmetry of this method's arguments (winning team first), we will want to create two
    sets of features for each match-up,
    - one where we list the winning team's features first, followed by the relative performance of the winning team
      vs. the losing team, followed by the losing team's features;
    - a second where we list the losing team's features first, followed by the relative performance of the losing
      team vs. the winning team, followed by the winning team's features. 
    
    We label the first set of feature with a 1 (first team won) and the second with a 0 (first team lost).
    """
    def get_match_up_features_and_labels(self, w_team_id: int, l_team_id: int):
        w_team_seed_position = self.tournament.seeds[w_team_id].position
        l_team_seed_position = self.tournament.seeds[l_team_id].position

        w_team_average_points_allowed = self.regular_season.get_average_points_allowed(w_team_id)
        l_team_average_points_allowed = self.regular_season.get_average_points_allowed(l_team_id)

        w_team_average_points_scored = self.regular_season.get_average_points_scored(w_team_id)
        l_team_average_points_scored = self.regular_season.get_average_points_scored(l_team_id)

        match_up_features = [[w_team_seed_position, w_team_average_points_allowed, w_team_average_points_scored,
                              l_team_seed_position, l_team_average_points_allowed, l_team_average_points_scored],
                             [l_team_seed_position, l_team_average_points_allowed, l_team_average_points_scored,
                              w_team_seed_position, w_team_average_points_allowed, w_team_average_points_scored]]
        match_up_labels = [1, 0]

        return match_up_features, match_up_labels

    """
    Returns features for the specified match-up: the output is an array of length 2 containing the features for the
    [[team_1_id, team_2_id], [team_2_id, team_1_id]] views of the match-up. 
    
    Although we call get_match_up_features_and_labels which relies on having the winning team's ID specified as its
    first argument, it does not matter in this case where we ignore the labels (who actually won). We're solely
    interested in the match-up features, and we return the two visions of the match-up: team_1 vs. team_2 and
    team_2 vs. team_1.
    """
    def get_match_up_features(self, team_1_id: int, team_2_id: int):
        return self.get_match_up_features_and_labels(team_1_id, team_2_id)[0]

    """
    Returns a labelled data set using the actual tournament games that occurred that season. The output is a tuple
    of two arrays, each of length two times the number of tournament games that year:
    - features first;
    - labels second.
    
    There are 60+ tournament games each season that we can learn from. We concatenate the match-up features
    and labels for each of those games (each separately).
    """
    def get_season_features_and_labels(self):
        season_features = []
        season_labels = []
        for game in self.tournament.tournament_games:
            # Order of arguments matter: winning team first, then losing team.
            match_up_features, match_up_labels = self.get_match_up_features_and_labels(game.w_team_id, game.l_team_id)
            season_features.extend(match_up_features)
            season_labels.extend(match_up_labels)
        return season_features, season_labels

    """
    Returns predictions for each of the potential tournament match-up for this season's NCAA tournament. 
    
    Because a match-up can be viewed as "team_1 vs. team_2" or "team_2 vs. team_1", and the predictions for these
    two point of views always sum up to 1 (no draws), we only output predictions for "team_1 vs. team_2" views, where
    the team_1's ID is strictly smaller than team_2's ID. That rule is enforced at the Sample class level.
    """
    def predict(self, classifier: Classifier) -> [Sample]:
        # Sorted array (ascending) of IDs of teams which participate to this season's NCAA tournament.
        tournament_teams_ids: [int] = self.tournament.team_ids

        """
        Go through the upper triangular matrix (without the diagonal: teams don't play themselves!) and predict the
        winning probability of team_1 vs. team_2. Number of match-ups: n * (n - 1) / 2, where n = number of teams. 
        """

        samples: [Sample] = []
        for idx_1, team_1_id in enumerate(tournament_teams_ids):
            # No need to enumerate for second team, we only use that team's ID.
            for team_2_id in tournament_teams_ids[idx_1+1:]:
                samples.append(self.get_sample(team_1_id, team_2_id))

        classes_probabilities = classifier.predict_proba(samples)
        for idx, sample in enumerate(samples):
            sample.win_p = classes_probabilities[idx][1]

        return samples

    """
    Get sample for team_1 vs. team_2 match-up.
    """
    def get_sample(self, team_1_id: int, team_2_id: int):
        expected_outcome = self.tournament.get_expected_outcome(team_1_id, team_2_id)
        match_up_features = self.get_match_up_features(team_1_id, team_2_id)

        return Sample(team_1_id, team_2_id, match_up_features[0], expected_outcome)

    @property
    def year(self):
        return self.regular_season.year

    """
    Returns the classifier based on seeds heuristics.
    """
    def get_seeds_based_classifier(self) -> Classifier:
        return SeedsBasedClassifier(self.tournament.seeds)


class RegularSeason:
    """
    A regular season contains information about the games played prior to the NCAA tournament: all division 1
    games included, even those involving teams which didn't make it to the tournament that year.

    The day zero serves as an origin date specific to that season and allows us to refer to commonly refer to day
    numbers for all seasons.
    """
    def __init__(self, year: int, day_zero: str, regular_season_games: [Game]):
        self.year: int = year
        self.day_zero: str = day_zero
        self.regular_season_games: [Game] = regular_season_games

        self.total_points_allowed: Dict[int, int] = {}
        self.total_points_scored: Dict[int, int] = {}
        self.number_games_played: Dict[int, int] = {}

        for game in self.regular_season_games:
            w_team_id = game.w_team_id
            l_team_id = game.l_team_id

            if w_team_id not in self.total_points_allowed:
                self.total_points_allowed[w_team_id] = 0
            if w_team_id not in self.total_points_scored:
                self.total_points_scored[w_team_id] = 0
            if w_team_id not in self.number_games_played:
                self.number_games_played[w_team_id] = 0

            if l_team_id not in self.total_points_allowed:
                self.total_points_allowed[l_team_id] = 0
            if l_team_id not in self.total_points_scored:
                self.total_points_scored[l_team_id] = 0
            if l_team_id not in self.number_games_played:
                self.number_games_played[l_team_id] = 0

            self.total_points_allowed[w_team_id] += game.l_score
            self.total_points_allowed[l_team_id] += game.w_score
            self.total_points_scored[w_team_id] += game.w_score
            self.total_points_scored[l_team_id] += game.l_score
            self.number_games_played[w_team_id] += 1
            self.number_games_played[l_team_id] += 1

        self.average_points_allowed: Dict[int, float] = {}
        for team_id, total_points in self.total_points_allowed.items():
            self.average_points_allowed[team_id] = total_points / self.number_games_played[team_id]

        self.average_points_scored: Dict[int, float] = {}
        for team_id, total_points in self.total_points_scored.items():
            self.average_points_scored[team_id] = total_points / self.number_games_played[team_id]

    def get_average_points_allowed(self, team_id: int) -> float:
        return self.average_points_allowed[team_id]

    def get_average_points_scored(self, team_id: int) -> float:
        return self.average_points_scored[team_id]

    def get_record(self, team_id: int) -> float:
        assert type(team_id) == int, f"Team ID {team_id} is not an integer."

        won_games = 0
        lost_games = 0
        for game in self.regular_season_games:
            if game.w_team_id == team_id:
                won_games += 1
            elif game.l_team_id == team_id:
                lost_games += 1
        assert won_games + lost_games != 0, f"No game records for team {team_id} during the {self.year-1}-{self.year}" \
                                            f" regular season."
        return won_games / (won_games + lost_games)
