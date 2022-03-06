class Team:
    def __init__(self, team_id: str, team_name: str, first_d1_season: int, last_d1_season: int):
        self.id = team_id
        self.name = team_name
        self.first_d1_season = first_d1_season
        self.last_d1_season = last_d1_season