from predictor import Predictor
from teams import Team


class Tournament:
    def __init__(self, teams: [Team]):
        self.teams = teams
        self.predictor = Predictor()

    def predict_match_up(self, year: int, team_1: Team, team_2: Team) -> float:
        return self.predictor.predict(year, team_1, team_2)