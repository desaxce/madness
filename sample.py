from math import log, inf


class Sample:
    """
    A sample wraps the features and the label of a specific match-up team_1 vs. team_2,
    with team_1_id < team_2_id.

    The label is the actual outcome of the game if it took place.

    Sample instances should always be looked at in the context of a specific year.
    """

    def __init__(self, team_1_id: int, team_2_id, features: [float], label: int = -1):
        assert team_1_id < team_2_id, f"Cannot create sample for the match-up {team_1_id} vs. {team_2_id}, " \
                                      f"because {team_1_id} > {team_2_id}."

        self.team_1_id: int = team_1_id
        self.team_2_id: int = team_2_id
        self.features: [float] = features

        # If team_1 won, label is 1. If team_1 lost, 0. Else, -1.
        self.label: int = label

        # Private properties, not initially written on a sample.
        self._win_p: float = 0.5
        self.score: float = inf

    @property
    def win_p(self):
        return self._win_p

    @win_p.setter
    def win_p(self, value: float):
        self._win_p = value
        self.score: float = 0
        if self.label == 1:
            self.score = -1 * log(self._win_p)
        elif self.label == 0:
            self.score = -1 * log(1 - self._win_p)


