import csv
from collections import Counter

from conftest import PROJECT_DIR
from src.data import load_premier_league, win_matrix

DATA = PROJECT_DIR / "data" / "premier-league-matches.csv"


def test_loader_matches_independent_parse():
    names, winners, losers, info = load_premier_league(DATA, n_games=2500)
    assert info["n_games"] == len(winners) == len(losers) == 2500
    assert names == sorted(names) and len(set(names)) == len(names) == info["n_teams"]
    assert (winners != losers).all()

    # Independent parse with the csv module: last 2500 decisive rows in file order.
    with open(DATA, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["FTR"] != "D"][-2500:]
    expected_wins = Counter(r["Home"] if r["FTR"] == "H" else r["Away"] for r in rows)
    expected_losses = Counter(r["Away"] if r["FTR"] == "H" else r["Home"] for r in rows)
    got_wins = Counter(names[i] for i in winners)
    got_losses = Counter(names[i] for i in losers)
    assert got_wins == expected_wins and got_losses == expected_losses
    assert info["first_date"] == rows[0]["Date"] and info["last_date"] == rows[-1]["Date"]


def test_win_matrix_counts():
    wins = win_matrix([0, 0, 1, 2], [1, 1, 0, 0], 3)
    assert wins[0, 1] == 2 and wins[1, 0] == 1 and wins[2, 0] == 1
    assert wins.sum() == 4 and wins.diagonal().sum() == 0
