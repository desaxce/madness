class AbsoluteFeature:
    """
    Represents a feature intrinsic to each team and not related to its opponent.

    E.g. team ID, reg. season avg pts allowed, reg. season game records, etc.
    """
    def __init__(self, team_1_value: float, team_2_value: float):
        self.value_1: float = team_1_value
        self.value_2: float = team_2_value


class RelativeFeature:
    """
    Represents a feature tied to a specific match-up team_1 vs. team_2.
    Stores both "visions" of the match-up.
    """
    def __init__(self, team_1_vs_team_2_value: float, team_2_vs_team_1_value: float):
        self.value_1 = team_1_vs_team_2_value
        self.value_2 = team_2_vs_team_1_value
