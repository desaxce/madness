from typing import Dict

from feature import AbsoluteFeature, Feature, RelativeFeature
from game import Game
from seed import Seed
from teams import MatchUp


class Tournament:
    """
    Represents a single NCAA tournament and contains:
    - the initial seeds;
    - the games played.
    """

    def __init__(self, year: int, tournament_games: [Game], region_w, region_x, region_y, region_z, seeds: [Seed], rankings: Dict):
        self.year: int = year
        self.tournament_games = tournament_games
        self.team_ids = sorted(list(set([game.l_team_id for game in self.tournament_games]) \
                                    | set([game.w_team_id for game in self.tournament_games])))

        self.expected_outcomes: dict[str, int] = {}
        for game in tournament_games:
            self.expected_outcomes[str(game)] = game.outcome()

        self.seeds = {}
        for seed in seeds:
            self.seeds[seed.team_id] = seed

        # Key is system name, value is a Dict[team_id, rank].
        self.rankings = rankings

    def get_seeds_positions(self, match_up: MatchUp) -> Feature:
        return AbsoluteFeature(self.seeds[match_up.team_1_id].position,
                               self.seeds[match_up.team_2_id].position)

    def get_ranking_diff(self, match_up: MatchUp) -> Feature:
        # No ranking data prior to 2003.
        if self.year < 2003:
            ranking_diff = 4 * (self.seeds[match_up.team_1_id].position - self.seeds[match_up.team_2_id].position)
        else:
            system_name = "MOR"
            assert system_name in self.rankings, f"No {system_name} rankings for year {self.year}"

            assert match_up.team_1_id in self.rankings[system_name], f"Team {match_up.team_1_id} is not part of " \
                                                                     f"the {system_name} ranking for year {self.year}"
            team_1_rank = self.rankings[system_name][match_up.team_1_id]

            assert match_up.team_2_id in self.rankings[system_name], f"Team {match_up.team_2_id} is not part of " \
                                                                     f"the {system_name} ranking for year {self.year}"
            team_2_rank = self.rankings[system_name][match_up.team_2_id]

            ranking_diff = team_1_rank - team_2_rank

        return RelativeFeature(ranking_diff, -1 * ranking_diff)

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
