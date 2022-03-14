class Game:
    def __init__(self, year: int, day_num: int, w_team_id: int, w_score: int,
                 l_team_id: int, l_score: int, w_loc, num_ot, w_fgm, w_fga, w_fgm3,  w_fga3,
                 w_ftm, w_fta, w_or, w_dr, w_ast, w_to, w_stl, w_blk, w_pf, l_fgm, l_fga, l_fgm3,  l_fga3,
                 l_ftm, l_fta, l_or, l_dr, l_ast, l_to, l_stl, l_blk, l_pf):
        self.year: int = year
        self.day_num: int = day_num
        self.w_team_id: int = w_team_id
        self.w_score: int = w_score
        self.l_team_id: int = l_team_id
        self.l_score: int = l_score
        self.w_loc = w_loc
        self.num_ot = num_ot
        self.is_w_team_smallest_id = self.w_team_id < self.l_team_id
        
        # Winning team stats.
        self.w_fgm: int = w_fgm
        self.w_fga: int = w_fga
        self.w_fgm3: int = w_fgm3
        self.w_fga3: int = w_fga3
        self.w_ftm: int = w_ftm
        self.w_fta: int = w_fta
        self.w_or: int = w_or
        self.w_dr: int = w_dr
        self.w_ast: int = w_ast
        self.w_to: int = w_to
        self.w_stl: int = w_stl
        self.w_blk: int = w_blk
        self.w_pf: int = w_pf

        # Losing team stats.
        self.l_fgm: int = l_fgm
        self.l_fga: int = l_fga
        self.l_fgm3: int = l_fgm3
        self.l_fga3: int = l_fga3
        self.l_ftm: int = l_ftm
        self.l_fta: int = l_fta
        self.l_or: int = l_or
        self.l_dr: int = l_dr
        self.l_ast: int = l_ast
        self.l_to: int = l_to
        self.l_stl: int = l_stl
        self.l_blk: int = l_blk
        self.l_pf: int = l_pf

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