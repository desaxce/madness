from game import Game


class Tournament:
    """
    Represents a single NCAA tournament and contains:
    - the initial seeds;
    - the games played.
    """

    def __init__(self, tournament_games: [Game], region_w, region_x, region_y, region_z):
        self.tournament_games = tournament_games
        self.team_ids = sorted(list(set([game.l_team_id for game in self.tournament_games]) \
                                    | set([game.w_team_id for game in self.tournament_games])))

        self.expected_outcomes: dict[str, int] = {}
        for game in tournament_games:
            self.expected_outcomes[str(game)] = game.outcome()

    """
    Returns the expected outcome of a potential match-up:
    - 1 if team_1 won;
    - 0 if team_1 lost;
    - -1 if the match-up didn't take place during that tournament. 
    
    This method expects team_1_id < team_2_id, because expected outcomes are keyed with the concatenation of the
    lowest team ID followed by the largest team ID. 
    """

    def get_expected_outcome(self, team_1_id, team_2_id):
        return self.expected_outcomes.get(f"{team_1_id}_{team_2_id}", -1)
