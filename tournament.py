from game import Game


class Tournament:
    def __init__(self, tournament_games: [Game], region_w, region_x, region_y, region_z):
        self.tournament_games = tournament_games