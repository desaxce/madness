class Team:
    def __init__(self, team_id: int, team_name: str, first_d1_season: int, last_d1_season: int):
        # TODO: Change the team_id to an integer to allow for comparisons.
        self.id: int = team_id
        self.name = team_name
        self.first_d1_season = first_d1_season
        self.last_d1_season = last_d1_season