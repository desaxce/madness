from typing import Dict

from feature import AbsoluteFeature, RelativeFeature, Feature
from game import Game
from classifier import Classifier, SeedsBasedClassifier
from sample import Sample
from teams import Team, MatchUp
from tournament import Tournament
from seed import Seed

from collections import defaultdict


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

    def __init__(self, year: int, day_zero: str, regular_season_games: [Game], rankings: Dict,
                 tournament_games: [Game], region_w: str, region_x: str, region_y: str, region_z: str, seeds: [Seed],
                 teams: [Team]):
        self.qualified_teams_ids: [str] = sorted([seed.team_id for seed in seeds])
        self.regular_season: RegularSeason = RegularSeason(year, day_zero, regular_season_games, self.qualified_teams_ids)
        self.tournament: Tournament = Tournament(year, tournament_games, region_w, region_x, region_y, region_z, seeds, rankings)
        self.teams: [Team] = teams

    """
    Returns float features for the specified match-up.
    
    We create two sets of features, one for each "vision" of the match-up:
    - team_1 vs. team_2;
    - team_2 vs. team_1.
    """

    def get_match_up_features(self, match_up: MatchUp):
        features: [Feature] = self.build_features(match_up)

        vision_1_features = []
        vision_2_features = []
        for feature in features:
            vision_1_features.extend(feature.vision_1())
            vision_2_features.extend(feature.vision_2())

        return [vision_1_features, vision_2_features]

    """
    Builds feature objects for a match-up.
    
    The features are of two kinds:
    - either they relate to a single team (absolute);
    - or involve the relative performance of the two matched up teams (relative).
<<<<<<< HEAD
    """

    def build_features(self, match_up: MatchUp) -> [Feature]:
        absolute_features: [Feature] = self.build_absolute_features(match_up)
        relative_features: [Feature] = self.build_relative_features(match_up)
        return absolute_features + relative_features

    """
    Relative features for a potential match-up in this season's NCAA tournament. 
    """

    def build_relative_features(self, match_up: MatchUp) -> [RelativeFeature]:
        ranking_diff_feature = self.tournament.get_ranking_diff(match_up)
        bracket_position_feature = self.tournament.get_bracket_positions(match_up)
        return [ranking_diff_feature]

    """
    Absolute features for a potential match-up in this season's NCAA tournament. 
    """

    def build_absolute_features(self, match_up: MatchUp) -> [AbsoluteFeature]:
        seed_position_feature = self.tournament.get_seeds_positions(match_up)
        average_points_allowed_feature = self.regular_season.get_average_points_allowed(match_up)
        average_points_scored_feature = self.regular_season.get_average_points_scored(match_up)
        adjusted_win_pct_feature = self.regular_season.get_adjusted_win_pct(match_up)

        return [seed_position_feature, adjusted_win_pct_feature]

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
            match_up_features = self.get_match_up_features(MatchUp(game.w_team_id, game.l_team_id))
            season_features.extend(match_up_features)
            season_labels.extend([1, 0])
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
            for team_2_id in tournament_teams_ids[idx_1 + 1:]:
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
        match_up_features = self.get_match_up_features(MatchUp(team_1_id, team_2_id))

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

    def __init__(self, year: int, day_zero: str, regular_season_games: [Game], qualified_teams_ids: [str]):
        self.year: int = year
        self.day_zero: str = day_zero
        self.regular_season_games: [Game] = regular_season_games
        self.qualified_teams_ids: [str] = qualified_teams_ids

        self.total_points_allowed: Dict[int, int] = defaultdict(lambda: 0)
        self.total_points_scored: Dict[int, int] = defaultdict(lambda: 0)
        self.number_games_played: Dict[int, int] = defaultdict(lambda: 0)
        self.adjusted_nb_wins: Dict[int, float] = defaultdict(lambda: 0)

        for game in self.regular_season_games:
            w_team_id = game.w_team_id
            l_team_id = game.l_team_id
            location = game.w_loc

            # Only consider regular season games involving tournament teams.
            if w_team_id in self.qualified_teams_ids and l_team_id in self.qualified_teams_ids:
                self.total_points_allowed[w_team_id] += game.l_score
                self.total_points_allowed[l_team_id] += game.w_score
                self.total_points_scored[w_team_id] += game.w_score
                self.total_points_scored[l_team_id] += game.l_score
                self.number_games_played[w_team_id] += 1
                self.number_games_played[l_team_id] += 1

                if location == "H":
                    self.adjusted_nb_wins[w_team_id] += 0.6
                elif location == "A":
                    self.adjusted_nb_wins[w_team_id] += 1.4
                elif location == "N":
                    self.adjusted_nb_wins[w_team_id] += 1

        self.average_points_allowed: Dict[int, float] = {}
        for team_id, total_points in self.total_points_allowed.items():
            self.average_points_allowed[team_id] = total_points / self.number_games_played[team_id]

        # In case there is a qualified team which didn't play any other qualified teams during
        # the regular season, we return an average number of points. Maybe try -1.
        assert len(self.average_points_allowed) > 0, f"No teams played in {self.year}"
        teams_average_points_allowed = sum(self.average_points_allowed.values()) / len(self.average_points_allowed)
        self.average_points_allowed = defaultdict(lambda: teams_average_points_allowed, self.average_points_allowed)

        self.average_points_scored: Dict[int, float] = {}
        for team_id, total_points in self.total_points_scored.items():
            self.average_points_scored[team_id] = total_points / self.number_games_played[team_id]

        # Same for points scored.
        teams_average_points_scored = sum(self.average_points_scored.values()) / len(self.average_points_scored)
        self.average_points_scored = defaultdict(lambda: teams_average_points_scored, self.average_points_scored)

        self.adjusted_win_pct = {}
        for team_id, number_games_played in self.number_games_played.items():
            self.adjusted_win_pct[team_id] = self.adjusted_nb_wins[team_id] / number_games_played

        # Same for adjusted win percentage.
        teams_avg_adjusted_win_pct = sum(self.adjusted_win_pct.values()) / len(self.adjusted_win_pct)
        self.adjusted_win_pct = defaultdict(lambda: teams_avg_adjusted_win_pct, self.adjusted_win_pct)

    def get_average_points_allowed(self, match_up: MatchUp) -> Feature:
        return AbsoluteFeature(self.average_points_allowed[match_up.team_1_id],
                               self.average_points_allowed[match_up.team_2_id])

    def get_average_points_scored(self, match_up: MatchUp) -> Feature:
        return AbsoluteFeature(self.average_points_scored[match_up.team_1_id],
                               self.average_points_scored[match_up.team_2_id])

    def get_adjusted_win_pct(self, match_up: MatchUp) -> Feature:
        return AbsoluteFeature(self.adjusted_win_pct[match_up.team_1_id],
                               self.adjusted_win_pct[match_up.team_2_id])

    def get_record(self, team_id: int) -> float:
        assert type(team_id) == int, f"Team ID {team_id} is not an integer."

        won_games = 0
        lost_games = 0
        for game in self.regular_season_games:
            if game.w_team_id == team_id:
                won_games += 1
            elif game.l_team_id == team_id:
                lost_games += 1
        assert won_games + lost_games != 0, f"No game records for team {team_id} during the {self.year - 1}-{self.year}" \
                                            f" regular season."
        return won_games / (won_games + lost_games)
