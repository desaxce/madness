from typing import Dict

from game import Game
from prediction import Prediction, Classifier, NeuralNetworkClassifier, FiftyFiftyClassifier, SeedsBasedClassifier
from sample import Sample
from teams import Team
from tournament import Tournament
from seed import Seed
from statistics import mean

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler


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
    the team_1's ID is strictly smaller than team_2's ID. That rule is enforced at the Prediction class level.
    """
    def predict(self, classifier: Classifier) -> [Prediction]:
        predictions: [Prediction] = []

        # Sorted array (ascending) of IDs of teams which participate to this season's NCAA tournament.
        tournament_teams_ids: [int] = self.tournament.team_ids

        """
        Go through the upper triangular matrix (without the diagonal: teams don't play themselves!) and predict the
        winning probability of team_1 vs. team_2. 
        
        If n is the number of tournament teams (may change each year, depending on the number of play-in games), 
        we generate n * (n - 1) / 2 predictions.
        """
        for idx_1, team_1_id in enumerate(tournament_teams_ids):
            # No need to enumerate for second team, we only use that team's ID.
            for team_2_id in tournament_teams_ids[idx_1+1:]:
                expected_outcome = self.tournament.get_expected_outcome(team_1_id, team_2_id)
                prediction = self.predict_match_up(classifier, team_1_id, team_2_id, expected_outcome)
                predictions.append(prediction)

        return predictions

    """
    Predicts a single match-up winning probability for the team with the smallest ID (team_1). 
    
    The argument team_1_id needs to be smaller than team_2_id.
    
    The method takes care of running predictions for both visions of the match-up, to validate the approach. 
    But it always outputs the winning probability from the vision team_1 vs. team_2. 
    
    This method is not optimal as it performs a prediction for a single match-up, when we would want to predict for
    a vector of match-ups simultaneously. If we do get confirmation that the first vision (team_1 vs. team_2) winning
    probability is equal to the second vision (team_2 vs. team_1) losing probability (algorithm symmetry), then we'll
    cut the number of calls to predict_proba by 2. Then we will bring the number of calls to predict_proba to 1
    per season.
    """
    def predict_match_up(self, classifier: Classifier, team_1_id: int, team_2_id: int, expected_outcome: int) -> Prediction:
        assert team_1_id != team_2_id, f"Cannot predict outcome of {team_1_id} vs. {team_2_id}, they're the " \
                                       f"same team"
        match_up_features = self.get_match_up_features(team_1_id, team_2_id)
        match_up_samples = [Sample(team_1_id, team_2_id, match_up_features[0])]
        predictions = classifier.predict_proba(match_up_samples)

        # First view is when we consider the team_1 vs. team_2 features: we retrieve the winning probability of team_1.
        first_view_win_prediction = predictions[0][1]

        # Second view is when we consider the team_2 vs. team_1 features: we retrieve the losing probability of team_2.
        # Ideally, the two predictions exactly match.
        # second_view_loss_prediction = predictions[1][0]

        return Prediction(self.year, team_1_id, team_2_id, first_view_win_prediction, expected_outcome)

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


class Span:
    """
    A span wraps a range of seasons, usually consecutive.
    Spans allow for manipulating several years of game data.

    To most common use of spans is to create two Span instances:
    - the first to learn the links between regular season and tournament match-ups;
    - the second to score the validity of those "learned" links.

    E.g. One span from 1985 to 2015 (included) serves as training data.
         A second span from 2016 to 2021 (included, but without 2020) serves as a
         validation data set.
    """

    def __init__(self, seasons: [Season]):
        self.seasons: [Season] = seasons

    """
    The train API lets users fit a Multi-Layer Perceptron classifier (using a logarithmic
    loss function) based on the outcomes of tournament games of each season in the span. 
    
    Each season brings its share of match-ups features based on the tournament games that took
    place that season: the outcome of those games is the label to learn and that we want to
    predict accurately in the end. It's 1 (resp. 0) if the first (resp. second) team wins. 
    
    Careful, the order of teams features matters.
     
    By concatenating all the seasons' features together (and similarly for all the labels), we
    can fit a classifier and return it for prediction purposes.
    """
    def train(self, max_iter: int = 1000) -> Classifier:
        span_features, span_labels = [], []

        # Concatenate each season's features and labels.
        for season in self.seasons:
            season_features, season_labels = season.get_season_features_and_labels()
            span_features.extend(season_features)
            span_labels.extend(season_labels)

        # All layers with the same size
        layer_size = len(span_features[0])
        print(len(span_features))
        print(len(span_features[0]))

        # Two layers for now.
        hidden_layer_sizes = (layer_size, layer_size * 2, layer_size * 3, layer_size * 2, layer_size)

        mlp_classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)

        # TODO: Scale features for faster convergence.
        # TODO: Monitor the learning part with loading bar for Jupyter notebook.
        mlp_classifier.fit(span_features, span_labels)

        return NeuralNetworkClassifier(mlp_classifier)

    """
    The predict API relies on a dictionary of classifiers indexed by the season's year (a classifier per season)
    to give predictions. If a specific season doesn't have an entry in that dictionary, we use the 50/50 classifier. 
    
    Be careful to always provide a classifier which was fitted on a span which does
    not overlap with the span you predict on. Otherwise, your results may be biased towards
    your training seasons.
    """
    def predict(self, classifiers: Dict = {}) -> Dict:
        # Map of season's year to predictions.
        span_predictions: Dict = {}

        for season in self.seasons:
            season_classifier = classifiers.get(season.year, FiftyFiftyClassifier())
            season_predictions = season.predict(season_classifier)
            span_predictions[season.year] = season_predictions

        return span_predictions

    """
    Returns exactly each season's seeds based classifier.
    """
    def get_seasons_seeds_based_classifiers(self) -> Dict[int, SeedsBasedClassifier]:
        seasons_seeds_based_classifiers: Dict[int, SeedsBasedClassifier] = {}

        for season in self.seasons:
            seasons_seeds_based_classifiers[season.year] = season.get_seeds_based_classifier()

        return seasons_seeds_based_classifiers

    """
    Returns a dictionary mapping each season to the same classifier, the one provided as input.
    """
    def build_seasons_classifiers_map(self, classifier: Classifier) -> Dict[int, Classifier]:
        seasons_classifiers: Dict[int, Classifier] = {}

        for season in self.seasons:
            seasons_classifiers[season.year] = classifier

        return seasons_classifiers

    """
    The score API takes in predictions for each of the seasons we want to score.
    
    It logs and returns each season's score and an average score for the span. 
    The output is a dictionary with the season's years as keys (-1 for the special case of the span score).
    """
    @staticmethod
    def score(span_predictions: Dict) -> Dict[int, float]:
        scores: Dict[int, float] = {}

        for year, season_predictions in span_predictions.items():
            relevant_predictions = []
            for prediction in season_predictions:
                if prediction.expected_outcome != -1:
                    relevant_predictions.append(prediction)

            relevant_predictions_scores = [prediction.score for prediction in relevant_predictions]
            scores[year] = mean(relevant_predictions_scores)

        span_score = mean(scores.values())
        scores[-1] = span_score

        return scores
