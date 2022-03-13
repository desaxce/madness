from statistics import mean
from typing import Dict

from sklearn.neural_network import MLPClassifier

from classifier import Classifier, FiftyFiftyClassifier, NeuralNetworkClassifier, SeedsBasedClassifier
from season import Season


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

        # Two layers for now.
        hidden_layer_sizes = (layer_size, layer_size)

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
                if prediction.label != -1:
                    relevant_predictions.append(prediction)

            relevant_predictions_scores = [prediction.score for prediction in relevant_predictions]
            scores[year] = mean(relevant_predictions_scores)

        span_score = mean(scores.values())
        scores["Average"] = span_score

        return scores