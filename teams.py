from feature import Feature, AbsoluteFeature


class Team:
    """
    Represents a division 1 team.
    """
    def __init__(self, team_id: int, team_name: str, first_d1_season: int = 0, last_d1_season: int = 0):
        self.id: int = team_id
        self.name: str = team_name
        self.first_d1_season: int = first_d1_season
        self.last_d1_season: int = last_d1_season


class MatchUp:
    """
    Represents a potential match-up of team_1 vs. team_2. No enforced rule on the teams IDs..
    """
    def __init__(self, team_1_id: int, team_2_id: int):
        self.team_1_id: int = team_1_id
        self.team_2_id: int = team_2_id

    def get_teams_ids_feature(self) -> Feature:
        return AbsoluteFeature(self.team_1_id, self.team_2_id)