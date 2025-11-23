"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """

        eps = 1e-8

        for i in range(self.lookback, len(self.price.index)):
            curr_date = self.price.index[i]

            try:
                past_date_idx = i - self.lookback
                past_prices = self.price[assets].iloc[past_date_idx]
                curr_prices = self.price[assets].iloc[i]
                momentum = (curr_prices / past_prices - 1).replace([np.inf, -np.inf], np.nan)
            except Exception:
                momentum = pd.Series(index=assets, data=np.nan)

            valid_mom = momentum.dropna()
            if len(valid_mom) == 0:
                selected = list(assets)
            else:
                selected = list(valid_mom.sort_values(ascending=False).iloc[:3].index)
                if len(selected) == 0:
                    selected = list(assets)

            window_returns = self.returns[assets].iloc[i - self.lookback : i]
            vol = window_returns.std(skipna=True)

            sel_vol = vol[selected].copy()
            sel_vol = sel_vol.fillna(np.nan)
            sel_vol[sel_vol <= 0] = np.nan
            if sel_vol.dropna().empty:
                w_selected = pd.Series(1.0 / len(selected), index=selected)
            else:
                sel_vol_filled = sel_vol.fillna(sel_vol.dropna().median())
                inv_vol = 1.0 / (sel_vol_filled + eps)
                w_selected = inv_vol / inv_vol.sum()

            row = pd.Series(0.0, index=self.price.columns)
            for a in selected:
                row[a] = float(w_selected.get(a, 0.0))

            s = row.sum()
            if s > 0:
                row = row / s
            else:
                avail = list(assets)
                ew = 1.0 / len(avail)
                for a in avail:
                    row[a] = ew

            self.portfolio_weights.loc[curr_date] = row
        
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
