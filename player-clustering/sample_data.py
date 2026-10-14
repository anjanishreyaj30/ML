"""
sample_data.py
--------------
Creates a realistic synthetic cricket dataset so you have something to demo
even before you find a real dataset.
"""

import numpy as np
import pandas as pd

FIRST_NAMES = [
    "Aarav", "Rohan", "Kabir", "Ishan", "Vikram", "Arjun", "Dev", "Karan",
    "Rahul", "Siddharth", "Aditya", "Manish", "Nikhil", "Pranav", "Yash",
    "Harsh", "Varun", "Aman", "Ritwik", "Sameer",
]
LAST_NAMES = [
    "Sharma", "Verma", "Patel", "Reddy", "Nair", "Iyer", "Khan", "Singh",
    "Gupta", "Mehta", "Rao", "Joshi", "Kulkarni", "Das", "Bose", "Chopra",
    "Malhotra", "Pillai", "Sinha", "Bhat",
]

COLUMNS = [
    "Player", "Matches", "Runs", "BattingAverage",
    "StrikeRate", "Wickets", "EconomyRate",
]


def generate_sample_data(n_players: int = 100, seed: int = 42) -> pd.DataFrame:
    """Build a DataFrame of n_players synthetic players."""
    rng = np.random.default_rng(seed)

    n_batters = int(round(n_players * 0.40))
    n_bowlers = int(round(n_players * 0.35))
    n_allrounders = n_players - n_batters - n_bowlers

    rows = []
    used_names = set()

    def new_name() -> str:
        while True:
            name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
            if name not in used_names:
                used_names.add(name)
                return name

    # ---------------- Batters ----------------
    for _ in range(n_batters):
        matches = int(rng.integers(20, 121))
        batting_avg = float(np.clip(rng.normal(38, 8), 18, 58))
        strike_rate = float(np.clip(rng.normal(132, 12), 95, 165))
        runs = int(batting_avg * matches * rng.uniform(0.75, 1.00))
        wickets = int(max(0, rng.normal(2, 3)))
        economy = 0.0 if wickets == 0 else round(float(np.clip(rng.normal(8.5, 1.0), 6.5, 11.0)), 2)
        rows.append([new_name(), matches, runs, round(batting_avg, 2),
                     round(strike_rate, 2), wickets, economy])

    # ---------------- Bowlers ----------------
    for _ in range(n_bowlers):
        matches = int(rng.integers(20, 121))
        wickets = int(np.clip(rng.normal(60, 25), 15, 140))
        economy = round(float(np.clip(rng.normal(7.8, 0.8), 5.8, 10.2)), 2)
        batting_avg = float(np.clip(rng.normal(14, 6), 3, 30))
        strike_rate = float(np.clip(rng.normal(85, 15), 50, 120))
        runs = int(max(0, rng.normal(180, 120)))
        rows.append([new_name(), matches, runs, round(batting_avg, 2),
                     round(strike_rate, 2), wickets, economy])

    # ---------------- All-rounders ----------------
    for _ in range(n_allrounders):
        matches = int(rng.integers(30, 121))
        batting_avg = float(np.clip(rng.normal(32, 7), 18, 50))
        strike_rate = float(np.clip(rng.normal(118, 12), 90, 150))
        runs = int(batting_avg * matches * rng.uniform(0.70, 0.95))
        wickets = int(np.clip(rng.normal(40, 18), 10, 100))
        economy = round(float(np.clip(rng.normal(8.0, 0.8), 6.0, 10.5)), 2)
        rows.append([new_name(), matches, runs, round(batting_avg, 2),
                     round(strike_rate, 2), wickets, economy])

    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Inject messiness so the cleaning step is meaningful
    for col in ["BattingAverage", "StrikeRate", "EconomyRate", "Wickets", "Runs"]:
        idx = rng.choice(df.index, size=3, replace=False)
        df.loc[idx, col] = np.nan

    duplicates = df.sample(3, random_state=seed)
    df = pd.concat([df, duplicates], ignore_index=True)

    return df


if __name__ == "__main__":
    data = generate_sample_data(100)
    data.to_csv("sample_players.csv", index=False)
    print(f"Wrote sample_players.csv with shape {data.shape}")