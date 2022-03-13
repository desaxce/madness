from abc import ABC, abstractmethod
from typing import Dict

from sklearn.neural_network import MLPClassifier

from sample import Sample
from seed import Seed


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
        return self.mlp_classifier.predict_proba(features)


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