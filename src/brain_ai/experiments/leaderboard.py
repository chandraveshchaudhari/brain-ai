from typing import List, Dict, Any


class Leaderboard:
    def __init__(self):
        self.entries: List[Dict[str, Any]] = []

    def add(self, result: Dict[str, Any]):
        self.entries.append(result)

    def summary(self) -> List[Dict[str, Any]]:
        return self.entries
