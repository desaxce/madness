class Seed:
    """
    Represents a single team's seed for a specific year.
    """
    def __init__(self, year: int, seed: str, team_id: int):
        self.year: int = year
        self.region: str = seed[0]
        self.position: int = int(seed[1:3])
        self.team_id: int = team_id