from season import Season
from teams import Team


class Predictor:
    """
        Predictor class containing main API for game predictions.
    """

    def __init__(self, seasons: [Season]):
        self.default_probability: float = 0.5
        self.seasons = seasons

    # def train(self):

    """
    Outputs the probability of team_1 winning vs. team_2 for a specific year.
    The two teams need to both be part of the NCAA tournament for that year.
    """
    def predict(self, year: int, team_1: Team, team_2: Team) -> float:
        return self.default_probability
