class Sample:
    """
    A sample wraps the features and the label of a specific match-up team_1 vs. team_2.

    When creating Sample objects for prediction purposes, the label value does not matter.
    """
    def __init__(self, team_1_id: int, team_2_id, features: [float], label: int = -1):
        self.team_1_id: int = team_1_id
        self.team_2_id: int = team_2_id
        self.features: [float] = features
        self.label: int = label