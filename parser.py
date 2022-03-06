import csv, logging

from teams import Team


class Parser:
    def __init__(self):
        self.filename = 'resources/MTeams.csv'
        self.logger = self._get_logger()

    @staticmethod
    def _get_logger():
        return logging.getLogger(__name__)

    def parse_teams(self) -> [Team]:
        teams: [Team] = []
        with open(self.filename) as teams_csv:
            csv_reader = csv.reader(teams_csv, delimiter=',')
            line_count: int = 0
            for row in csv_reader:
                if line_count == 0:
                    self.logger.info(f'Column names are {", ".join(row)}')
                else:
                    team_id: str = row[0]
                    team_name: str = row[1]
                    first_d1_season: int = int(row[2])
                    last_d1_season: int = int(row[3])
                    teams.append(Team(team_id, team_name, first_d1_season, last_d1_season))
                line_count += 1
            self.logger.info(f'Processed {line_count - 1} teams.')
        return teams
