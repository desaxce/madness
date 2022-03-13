from abc import ABC, abstractmethod
from math import log
from typing import Dict

from sklearn.neural_network import MLPClassifier

from sample import Sample
from seed import Seed


class Prediction:
    """
    Represents a single match-up prediction.
    The two teams IDs must follow the rule team_1_id < team_2_id.

    It should always be looked at in the context of a specific year.
    """

    def __init__(self, year: int, team_1_id: int, team_2_id: int, probability: float = 0.5, expected_outcome: int = -1):
        assert team_1_id < team_2_id, f"Cannot create prediction for the match-up {team_1_id} vs. {team_2_id}, " \
                                      f"because {team_1_id} > {team_2_id}."
        self.year: int = year
        self.team_1_id: int = team_1_id
        self.team_2_id: int = team_2_id
        self.probability: float = probability

        """
        We give the expected outcome (1 = team_1 won, 0 = team_1 lost) of that match-up to be able to score with only
        Prediction instances.
        
        For fictional match-ups that did not actually take place during a season's tournament, we give an expected
        outcome of -1. This does not impact scoring because we ignore such predictions. 
        """
        self.expected_outcome: int = expected_outcome

        """
        Score of that prediction. Defaults to 0 if game did not take place.
        """
        self.score: float = 0
        if self.expected_outcome == 1:
            self.score = -1 * log(self.probability)
        elif self.expected_outcome == 0:
            self.score = -1 * log(1 - self.probability)


class Classifier(ABC):

    """
    Returns an array class probabilities: because we have two classes (win and loss), each element of the output
    array is of size 2: [p, 1-p], where p is the probability of class 0 (loss).
    """
    @abstractmethod
    def predict_proba(self, samples: [Sample]):
        pass


class FiftyFiftyClassifier(Classifier):
    """
    Trivial classifier which gives a 50/50 chance of win vs. loss.
    """

    def predict_proba(self, samples: [Sample]):
        return [[0.5, 0.5]] * len(samples)


class NeuralNetworkClassifier(Classifier):
    """
    Wrapper classifier around an MLPClassifier.
    """
    def __init__(self, mlp_classifier: MLPClassifier):
        self.mlp_classifier: MLPClassifier = mlp_classifier

    def predict_proba(self, samples: [Sample]):
        features = [sample.features for sample in samples]
        return self.mlp_classifier.predict_proba([features[0], features[0]])


class SeedsBasedClassifier(Classifier):
    """
    Classifier using heuristics on seeds data to predict.

    Takes a dictionary as input, where a key is a team ID and the value the seed for that team.
    All values (Seed instances) in that dictionary should have the same year, otherwise the classifier has
    little meaning.
    """
    def __init__(self, seeds: Dict[int, Seed], spread: float = 0.4):
        self.seeds: Dict[int, Seed] = seeds
        self.spread: float = spread

    def predict_proba(self, samples: [Sample]):
        probabilities = []
        for sample in samples:
            team_1_seed_position: int = self.seeds[sample.team_1_id].position
            team_2_seed_position: int = self.seeds[sample.team_2_id].position

            seeds_difference = -1 * (team_1_seed_position - team_2_seed_position)
            team_1_win_probability = 0.5 + seeds_difference * (self.spread / 15)

            probabilities.append([1 - team_1_win_probability, team_1_win_probability])
