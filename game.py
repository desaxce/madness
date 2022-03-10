class Game:
    def __init__(self, year: int, day_num: int, w_team_id: int, w_score: int,
                 l_team_id: int, l_score: int, w_loc, num_ot):
        self.year: int = year
        self.day_num: int = day_num
        self.w_team_id: int = w_team_id
        self.w_score: int = w_score
        self.l_team_id: int = l_team_id
        self.l_score: int = l_score
        self.w_loc = w_loc
        self.num_ot = num_ot
        self.is_w_team_smallest_id = self.w_team_id < self.l_team_id

    """
    Concatenate team_1_id with team_2_id where team_1_id < team_2_id and use it as an identifier within a season's
    NCAA tournament. 
    
    You can not use this string representation across seasons, not even across regular season and tournament games of
    a same season. Otherwise, you are fine within a single season's NCAA tournament (no two teams play each other twice, 
    because NCAA tournament does not contain a group phase).
    """
    def __str__(self) -> str:
        team_1_id, team_2_id = (self.w_team_id, self.l_team_id) if self.is_w_team_smallest_id \
                          else (self.l_team_id, self.w_team_id)
        return f"{team_1_id}_{team_2_id}"

    """
    Returns the outcome of the game (win = 0, loss = 1) from the perspective of the team with smallest ID. 
    """
    def outcome(self) -> int:
        return 1 if self.is_w_team_smallest_id else 0