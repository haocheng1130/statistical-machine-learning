"""Premier League match data and two-player toy data sets.

Preprocessing of the match file:
  * draws (FTR == "D") are dropped, because the model only describes wins and losses;
  * the most recent `n_games` decisive matches are kept (the file is in date order);
  * teams that appear in those matches are indexed in alphabetical order;
  * each match becomes one (winner, loser) pair; home advantage is ignored.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_premier_league(path: Path, n_games: int = 2500):
    """Return (names, winners, losers, info) for the last `n_games` decisive matches."""
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["Date"])
    if not dates.is_monotonic_increasing:
        raise ValueError("expected the match file to be sorted by date")
    decisive = df[df["FTR"] != "D"].iloc[-n_games:]

    names = sorted(pd.concat([decisive["Home"], decisive["Away"]]).unique())
    index = {name: i for i, name in enumerate(names)}
    home = decisive["Home"].map(index).to_numpy()
    away = decisive["Away"].map(index).to_numpy()
    home_won = (decisive["FTR"] == "H").to_numpy()
    winners = np.where(home_won, home, away)
    losers = np.where(home_won, away, home)

    first, last = decisive["Date"].iloc[0], decisive["Date"].iloc[-1]
    in_span = df[(df["Date"] >= first) & (df["Date"] <= last)]
    info = {
        "n_games": int(len(decisive)),
        "n_teams": len(names),
        "first_date": first,
        "last_date": last,
        "draws_dropped_in_span": int((in_span["FTR"] == "D").sum()),
        "home_win_fraction": float(home_won.mean()),
    }
    return names, winners, losers, info


def win_matrix(winners, losers, n_players: int) -> torch.Tensor:
    """W[i, j] = number of games in which player i beat player j (float64)."""
    w = np.zeros((n_players, n_players))
    np.add.at(w, (np.asarray(winners), np.asarray(losers)), 1.0)
    return torch.tensor(w, dtype=torch.float64)


def two_player_wins(a_wins: int, b_wins: int) -> torch.Tensor:
    """Win matrix for two players A (index 0) and B (index 1)."""
    return torch.tensor([[0.0, a_wins], [b_wins, 0.0]], dtype=torch.float64)
