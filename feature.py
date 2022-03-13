class AbsoluteFeature:
    """
    Represents a feature intrinsic to a team and not related to its opponent.

    E.g. team ID, reg. season avg pts allowed, reg. season game records, etc.
    """
    def __init__(self, value):
        self.value = value


class RelativeFeature:
    """
    Represents a feature tied to a specific match-up team_1 vs. team_2.
    """
    def __init__(self, value):
        self.value = value
